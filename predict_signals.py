import glob
import itertools
import json
import sys
from pathlib import Path

import click
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from helpers.data import get_corpus_path, load_docs, get_doc_embeddings, get_paragraph_embeddings
from helpers.labeling import DiscourseSignalExtractor
from helpers.senses import get_bert_features, DiscourseSenseClassifier, DiscourseSenseEnsembleClassifier


def iter_documents_paragraphs(docs):
    def fmt(par, par_i):
        return {
            'doc_id': doc.doc_id,
            'paragraph_idx': par_i,
            'sentences': par,
        }

    for doc in docs:
        par = []
        par_i = 0
        for s in doc.sentences:
            if len(par) == 0 or (par[-1].tokens[-1].offset_end + 1 == s.tokens[0].offset_begin):
                par.append(s)
            else:
                yield fmt(par, par_i)
                par_i += 1
                par = [s]
        yield fmt(par, par_i)


class DiscourseSignalModel:
    def __init__(self, tokenizer, signal_model, sense_model_embed, sense_model, device='cpu'):
        self.tokenizer = tokenizer
        self.signal_model: DiscourseSignalExtractor = signal_model
        self.sense_model_embed = sense_model_embed
        self.sense_model = sense_model
        self.device = device

    @staticmethod
    def load_model(save_path, relation_type='altlex', device='cpu'):
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True, local_files_only=True)
        signal_model = DiscourseSignalExtractor.load_model(save_path, device=device)

        save_paths = glob.glob(save_path)
        sense_model_embed = AutoModel.from_pretrained("roberta-base", local_files_only=True)
        if len(save_paths) == 1:
            sense_model = DiscourseSenseClassifier.load(save_paths[0], relation_type=relation_type.lower())
        else:
            sense_model = DiscourseSenseEnsembleClassifier.load(save_paths, device, relation_type=relation_type.lower())
        sense_model_embed.to(device)
        sense_model.to(device)
        return DiscourseSignalModel(tokenizer, signal_model, sense_model_embed, sense_model, device)

    def predict(self, doc):
        try:
            sentence_embeddings = get_doc_embeddings(doc, self.tokenizer, self.sense_model_embed, device=self.device)
        except RuntimeError as e:
            sys.stderr.write(f"Error {doc.doc_id}: {e}")
            return []

        doc_signals = self.signal_model.predict(doc)
        for par_i, paragraph in enumerate(doc_signals):
            if paragraph['relations']:
                features = np.stack([get_bert_features(signal['tokens_idx'], sentence_embeddings)
                                     for signal in paragraph['relations']])
                features = torch.from_numpy(features).to(self.device)
                pred = self.sense_model.predict(features)

                relations = []
                for signal, coarse_class_i, coarse_class_i_prob, fine_class_i, fine_class_i_prob, coarse_class_i_prob_all in zip(
                        paragraph['relations'],
                        pred['coarse'], pred['probs_coarse'], pred['fine'], pred['probs_fine'],
                        pred['probs_coarse_all']):
                    relations.append({
                        'tokens_idx': signal['tokens_idx'],
                        'tokens': signal['tokens'],
                        'coarse': coarse_class_i,
                        'coarse_probs': round(coarse_class_i_prob, 4),
                        'fine': fine_class_i,
                        'fine_probs': round(fine_class_i_prob, 4),
                        'is_relation': (1.0 - coarse_class_i_prob_all[self.sense_model.label2id_coarse['None']]),
                    })
                doc_signals[par_i]['relations'] = relations
        return doc_signals

    def predict_paragraphs(self, paragraphs, is_relation_threshold=0.4):
        try:
            paragraph_embeddings = get_paragraph_embeddings(paragraphs, self.tokenizer, self.sense_model_embed,
                                                            device=self.device)
        except RuntimeError as e:
            sys.stderr.write(f"Error: {e}")
            return []

        doc_signals = self.signal_model.predict_paragraphs(paragraphs)
        for par_i, paragraph in enumerate(doc_signals):
            if paragraph['relations']:
                par_offset = paragraph['tokens_idx'][0]
                features = np.stack([get_bert_features([i - par_offset for i in signal['tokens_idx']],
                                                       paragraph_embeddings[par_i])
                                     for signal in paragraph['relations']])
                features = torch.from_numpy(features).to(self.device)
                pred = self.sense_model.predict(features)

                relations = []
                for signal, coarse_class_i, coarse_class_i_prob, fine_class_i, fine_class_i_prob, coarse_class_i_prob_all in zip(
                        paragraph['relations'],
                        pred['coarse'], pred['probs_coarse'], pred['fine'], pred['probs_fine'],
                        pred['probs_coarse_all']):
                    is_relation_prob = 1.0 - coarse_class_i_prob_all[self.sense_model.label2id_coarse['None']]
                    if is_relation_prob > is_relation_threshold:
                        relations.append({
                            'tokens_idx': signal['tokens_idx'],
                            'tokens': signal['tokens'],
                            'coarse': coarse_class_i,
                            'coarse_probs': round(coarse_class_i_prob, 4),
                            'fine': fine_class_i,
                            'fine_probs': round(fine_class_i_prob, 4),
                            'is_relation': is_relation_prob,
                        })
                paragraph['relations'] = relations
                yield paragraph


def filter_relation_connectives(paragraph):
    return paragraph


@click.command()
@click.argument('corpus')
@click.argument('save-path')
@click.option('-r', '--replace', is_flag=True)
@click.option('-o', '--output-path', default='-')
@click.option('-l', '--limit', default=0, type=int)
@click.option('-b', '--batch-size', default=32, type=int)
@click.option('-p', '--positives-only', is_flag=True)
@click.option('--is-relation-threshold', default=0.4, type=float)
@click.option('--sample-ratio', default=1.0, type=float)
@click.option('--filter-connectives', is_flag=True)
def main(corpus, save_path, replace, output_path, limit, batch_size, positives_only, is_relation_threshold,
         sample_ratio, filter_connectives):
    if output_path == '-':
        output = sys.stdout
    else:
        output_path = Path(output_path)
        if output_path.is_file() and output_path.stat().st_size > 100 and not replace:
            print('File already exists: Exit without writing.', file=sys.stderr)
            return
        else:
            output_path.parent.mkdir(exist_ok=True)
            output = output_path.open('w')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    signal_model = DiscourseSignalModel.load_model(save_path, device=device)

    corpus_path = get_corpus_path(corpus)
    paragraphs = filter(lambda p: 7 < sum(len(s.tokens) for s in p['sentences']) < 350,
                        iter_documents_paragraphs(load_docs(corpus_path, limit=limit, sample=sample_ratio)))
    while True:
        batch = list(itertools.islice(paragraphs, batch_size))
        if len(batch) == 0:
            break
        signals = signal_model.predict_paragraphs(batch, is_relation_threshold)
        if filter_connectives:
            signals = map(filter_relation_connectives, signals)
        if positives_only:
            signals = filter(lambda s: len(s['relations']) > 0, signals)
        for sent in signals:
            json.dump(sent, output)
            output.write('\n')
        output.flush()


if __name__ == '__main__':
    main()
