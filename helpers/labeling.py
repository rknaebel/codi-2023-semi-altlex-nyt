from collections import Counter

import torch
from torch.utils.data import Dataset

from helpers.data import load_docs, iter_document_paragraphs


def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(examples["tokens"],
                                 is_split_into_words=True, padding="max_length", return_tensors='pt')
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def get_explicit_label_mapping(doc):
    token_tag_map = {}
    for r in doc.relations:
        if r.is_explicit():
            for t_i, t in enumerate(r.conn.tokens):
                if t.idx not in token_tag_map:
                    if len(r.conn.tokens) == 1:
                        label = 'S-CONN'
                    else:
                        if t_i == 0:
                            label = 'B-CONN'
                        elif t_i == (len(r.conn.tokens) - 1):
                            label = 'E-CONN'
                        else:
                            label = 'I-CONN'
                    token_tag_map[t.idx] = label
    return token_tag_map


def get_altlex_label_mapping(doc):
    token_tag_map = {}
    for r in doc.relations:
        if r.type.lower().startswith('altlex'):
            for t_i, t in enumerate(r.conn.tokens):
                if t.idx not in token_tag_map:
                    if len(r.conn.tokens) == 1:
                        label = 'S-ALTLEX'
                    else:
                        if t_i == 0:
                            label = 'B-ALTLEX'
                        elif t_i == (len(r.conn.tokens) - 1):
                            label = 'E-ALTLEX'
                        else:
                            label = 'I-ALTLEX'
                    token_tag_map[t.idx] = label
    return token_tag_map


class ConnDataset(Dataset):

    def __init__(self, data_file, relation_type='explicit'):
        self.items = []
        self.labels = {'O': 0}

        for doc_i, doc in enumerate(load_docs(data_file)):
            if relation_type.lower() == 'explicit':
                label_mapping = get_explicit_label_mapping(doc)
            else:
                label_mapping = get_altlex_label_mapping(doc)
            for p_i, paragraph in enumerate(iter_document_paragraphs(doc)):
                tokens = []
                labels = []
                for sent in paragraph:
                    for tok in sent.tokens:
                        tokens.append(tok.surface)
                        label = label_mapping.get(tok.idx, 'O')
                        if label in self.labels:
                            label_id = self.labels[label]
                        else:
                            label_id = len(self.labels)
                            self.labels[label] = label_id
                        labels.append(label_id)
                if len(tokens) <= 2:
                    continue
                self.items.append({
                    'id': f"{doc_i}-{p_i}",
                    'tokens': tokens,
                    'tags': labels
                })

    def get_num_labels(self):
        return len(self.labels)

    def get_label_counts(self):
        return Counter(t for i in self.items for t in i['tags'])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    @staticmethod
    def get_collate_fn():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

        def collate(examples):
            examples = {
                'tokens': [example['tokens'] for example in examples],
                'tags': [example['tags'] for example in examples],
            }
            batch = tokenize_and_align_labels(tokenizer, examples)
            batch['labels'] = torch.LongTensor(batch['labels'])
            return batch

        return collate
