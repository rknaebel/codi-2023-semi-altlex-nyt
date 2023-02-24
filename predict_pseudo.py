import csv
import glob
import os
import sys
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel

from helpers.data import get_corpus_path, iter_document_paragraphs, load_docs, get_doc_embeddings
from helpers.senses import get_bert_features, DiscourseSenseClassifier, DiscourseSenseEnsembleClassifier


def decode_labels(tokens, labels, probs):
    conns = []
    for tok_i, (tok, label, prob) in enumerate(zip(tokens, labels, probs)):
        if label.startswith('S'):
            conns.append([(probs, tok)])
    conn_stack = []
    conn_cur = []
    for tok_i, (tok, label, prob) in enumerate(zip(tokens, labels, probs)):
        if label.startswith('B'):
            if conn_cur:
                conn_stack.append(conn_cur)
                conn_cur = []
            conn_cur.append((prob, tok))
        elif label.startswith('I'):
            if conn_cur:
                conn_cur.append((prob, tok))
            else:
                conn_cur = conn_stack.pop() if conn_stack else []
                conn_cur.append((prob, tok))
        elif label.startswith('E'):
            if conn_cur:
                conn_cur.append((prob, tok))
                conns.append(conn_cur)
            if conn_stack:
                conn_cur = conn_stack.pop()
            else:
                conn_cur = []
    return conns


# class EnsembleDiscourseLabelModel:


class DiscourseSignalExtractor:
    def __init__(self, tokenizer, signal_models, sense_model_embed, sense_model, device='cpu'):
        self.tokenizer = tokenizer
        self.signal_models = signal_models
        self.sense_model_embed = sense_model_embed
        self.sense_model = sense_model
        self.device = device

    @staticmethod
    def load_model(save_path, relation_type='altlex', device='cpu'):
        save_paths = glob.glob(save_path)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
        print("Load Signal Labeling Model(s)")
        signal_models = []
        for save_path in save_paths:
            label_save_path = os.path.join(save_path, f"best_model_{relation_type.lower()}_label")
            model = AutoModelForTokenClassification.from_pretrained(label_save_path)
            model.eval()
            model.to(device)
            signal_models.append(model)

        sense_model_embed = AutoModel.from_pretrained("roberta-base")
        if len(save_paths) == 1:
            sense_model = DiscourseSenseClassifier.load(save_paths[0], relation_type=relation_type.lower())
        else:
            sense_model = DiscourseSenseEnsembleClassifier.load(save_paths, device, relation_type=relation_type.lower())
        sense_model_embed.to(device)
        sense_model.to(device)
        return DiscourseSignalExtractor(tokenizer, signal_models, sense_model_embed, sense_model, device)

    def predict(self, doc):
        try:
            sentence_embeddings = get_doc_embeddings(doc, self.tokenizer, self.sense_model_embed, device=self.device)
        except RuntimeError as e:
            sys.stderr.write(f"Error {doc.doc_id}: {e}")
            return []

        document_relations = []
        for p_i, paragraph in enumerate(iter_document_paragraphs(doc)):
            tokens = [t for s in paragraph for t in s.tokens]
            input_strings = [t.surface for t in tokens]
            inputs = self.tokenizer(input_strings, truncation=True, is_split_into_words=True,
                                    padding="max_length", return_tensors='pt')
            _inputs = {k: v.to(self.device) for k, v in inputs.items()}
            probs, predictions = self.compute_ensemble_prediction(_inputs)
            word_ids = inputs.word_ids()
            predicted_token_class = [self.signal_models[0].config.id2label[t] for t in predictions.tolist()[0]]
            predicted_token_prob = probs.tolist()[0]
            word_id_map = []
            for i, wi in enumerate(word_ids):
                if wi is not None and (len(word_id_map) == 0 or (word_ids[i - 1] != wi)):
                    word_id_map.append(i)
            signals = decode_labels(tokens,
                                    [predicted_token_class[i] for i in word_id_map],
                                    [predicted_token_prob[i] for i in word_id_map])

            paragraph_relations = []
            for sent_i, sent in enumerate(paragraph):
                sentence_signals = [signal for signal in signals if signal[0][1].sent_idx == sent.tokens[0].sent_idx]
                if sentence_signals:
                    features = np.stack([get_bert_features([t.idx for i, t in signal], sentence_embeddings)]
                                        for signal in sentence_signals)
                    features = torch.from_numpy(features).to(self.device)
                    pred = self.sense_model.predict(features)

                    for signal, coarse_class_i, coarse_class_i_prob, fine_class_i, fine_class_i_prob in zip(
                            sentence_signals,
                            pred['coarse'], pred['probs_coarse'], pred['fine'], pred['probs_fine']):
                        paragraph_relations.append([
                            doc.doc_id, p_i, sent_i,
                            np.mean([p for p, t in signal]).round(4),
                            '-'.join(str(t.idx) for i, t in signal),
                            '-'.join(t.surface.lower() for i, t in signal),
                            coarse_class_i, round(coarse_class_i_prob, 4),
                            fine_class_i, round(fine_class_i_prob, 4)
                        ])
            document_relations.append(paragraph_relations)
        return document_relations

    def compute_ensemble_prediction(self, batch):
        predictions = []
        with torch.no_grad():
            for model in self.signal_models:
                outputs = model(**batch)
                predictions.append(F.softmax(outputs.logits, dim=-1))
        return torch.max(torch.mean(torch.stack(predictions), dim=0), dim=-1)


@click.command()
@click.argument('corpus')
@click.argument('save-path-a')
@click.argument('save-path-b')
@click.option('-r', '--replace', is_flag=True)
@click.option('-o', '--output-path', default='-')
@click.option('-l', '--limit', default=0, type=int)
def main(corpus, save_path_a, save_path_b, replace, output_path, limit):
    # relation_type = 'altlex'
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

    signal_model_a = DiscourseSignalExtractor.load_model(save_path_a, device=device)
    signal_model_b = DiscourseSignalExtractor.load_model(save_path_b, device=device)

    corpus_path = get_corpus_path(corpus)
    # save_paths = glob.glob(save_path)
    # label_save_path = os.path.join(save_paths[0], f"best_model_{relation_type.lower()}_label")
    # tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    # signal_model = AutoModelForTokenClassification.from_pretrained(label_save_path)
    # sense_model_embed = AutoModel.from_pretrained("roberta-base")
    # if len(save_paths) == 1:
    #     sense_model = DiscourseSenseClassifier.load(save_paths[0], relation_type=relation_type.lower())
    # else:
    #     sense_model = DiscourseSenseEnsembleClassifier.load(save_paths, device, relation_type=relation_type.lower())

    csv_out = csv.writer(output)

    # signal_model.to(device)
    # sense_model_embed.to(device)
    # sense_model.to(device)
    for doc_i, doc in enumerate(load_docs(corpus_path)):
        if limit and doc_i >= limit:
            break
        doc_relations_a = signal_model_a.predict(doc)
        print('DOC ID', doc.doc_id)
        print('- A ---')
        for par_relations in doc_relations_a:
            csv_out.writerows(par_relations)
        print('- B ---')
        doc_relations_b = signal_model_b.predict(doc)
        for par_relations in doc_relations_b:
            csv_out.writerows(par_relations)

        # try:
        #     sentence_embeddings = get_doc_embeddings(doc, tokenizer, sense_model_embed, device=device)
        # except RuntimeError as e:
        #     sys.stderr.write(f"Error {doc_i}: {e}")
        #     continue
        # for p_i, paragraph in enumerate(iter_document_paragraphs(doc)):
        #     tokens = [t for s in paragraph for t in s.tokens]
        #     input_strings = [t.surface for t in tokens]
        #     inputs = tokenizer(input_strings, truncation=True, is_split_into_words=True,
        #                        padding="max_length", return_tensors='pt')
        #     with torch.no_grad():
        #         _inputs = {k: v.to(device) for k, v in inputs.items()}
        #         logits = signal_model(**_inputs).logits
        #         logits = F.softmax(logits, dim=-1)
        #     probs, predictions = torch.max(logits, dim=2)
        #     word_ids = inputs.word_ids()
        #     predicted_token_class = [signal_model.config.id2label[t] for t in predictions.tolist()[0]]
        #     predicted_token_prob = probs.tolist()[0]
        #     word_id_map = []
        #     for i, wi in enumerate(word_ids):
        #         if wi is not None and (len(word_id_map) == 0 or (word_ids[i - 1] != wi)):
        #             word_id_map.append(i)
        #     signals = decode_labels(tokens,
        #                             [predicted_token_class[i] for i in word_id_map],
        #                             [predicted_token_prob[i] for i in word_id_map])
        #     for sent_i, sent in enumerate(paragraph):
        #         sentence_signals = [signal for signal in signals if signal[0][1].sent_idx == sent.tokens[0].sent_idx]
        #         if sentence_signals:
        #             features = np.stack([get_bert_features([t.idx for i, t in signal], sentence_embeddings)]
        #                                 for signal in sentence_signals)
        #             features = torch.from_numpy(features).to(device)
        #             pred = sense_model.predict(features)
        #
        #             for signal, coarse_class_i, coarse_class_i_prob, fine_class_i, fine_class_i_prob in zip(
        #                     sentence_signals,
        #                     pred['coarse'], pred['probs_coarse'], pred['fine'], pred['probs_fine']):
        #                 csv_out.writerow([
        #                     corpus, 'altlex',
        #                     doc.doc_id, p_i, sent_i,
        #                     np.mean([p for p, t in signal]).round(4),
        #                     '-'.join(str(t.idx) for i, t in signal),
        #                     '-'.join(t.surface.lower() for i, t in signal),
        #                     coarse_class_i, round(coarse_class_i_prob, 4),
        #                     fine_class_i, round(fine_class_i_prob, 4)
        #                     # ' '.join(t.surface for s in [s for s in doc.sentences if ])
        #                 ])
        output.flush()


if __name__ == '__main__':
    main()
