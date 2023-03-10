import glob
import json
import sys
from pathlib import Path

import click
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from helpers.data import get_corpus_path, load_docs, get_doc_embeddings
from helpers.labeling import DiscourseSignalExtractor
from helpers.senses import get_bert_features, DiscourseSenseClassifier, DiscourseSenseEnsembleClassifier


class DiscourseSignalModel:
    def __init__(self, tokenizer, signal_model, sense_model_embed, sense_model, device='cpu'):
        self.tokenizer = tokenizer
        self.signal_model = signal_model
        self.sense_model_embed = sense_model_embed
        self.sense_model = sense_model
        self.device = device

    @staticmethod
    def load_model(save_path, relation_type='altlex', device='cpu'):
        save_paths = glob.glob(save_path)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
        signal_model = DiscourseSignalExtractor.load_model(save_path, device=device)

        sense_model_embed = AutoModel.from_pretrained("roberta-base")
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


@click.command()
@click.argument('corpus')
@click.argument('save-path')
@click.option('-r', '--replace', is_flag=True)
@click.option('-o', '--output-path', default='-')
@click.option('-l', '--limit', default=0, type=int)
def main(corpus, save_path, replace, output_path, limit):
    if output_path == '-':
        output = sys.stdout
    else:
        output_path = Path(output_path)
        if output_path.is_file() and output_path.stat().st_size > 100 and not replace:
            sys.stderr.write('File already exists: Exit without writing.')
            return
        else:
            output_path.parent.mkdir(exist_ok=True)
            output = output_path.open('w')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    signal_model = DiscourseSignalModel.load_model(save_path, device=device)

    corpus_path = get_corpus_path(corpus)
    for doc_i, doc in enumerate(load_docs(corpus_path)):
        if limit and doc_i >= limit:
            break
        signals = signal_model.predict(doc)
        for s in signals:
            if len(s['relations']) > 0:
                json.dump(s, output)
                output.write('\n')
        output.flush()


if __name__ == '__main__':
    main()
