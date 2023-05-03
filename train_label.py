import json
import os
import sys
from pathlib import Path

import click
import evaluate
import numpy as np
import sklearn.model_selection
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification
from transformers import get_scheduler

from helpers.data import get_corpus_path, load_docs
from helpers.labeling import ConnDataset
from helpers.stats import print_metrics_results, print_final_results


# from discopy.utils import single_connectives, multi_connectives_first, multi_connectives, distant_connectives


def compute_loss(num_labels, weights, logits, labels, device):
    loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    return loss


def sort_select_items(items, limit_ratio):
    items_sort = np.argsort([p['prob'] for p in items])
    limit = int(len(items_sort) * limit_ratio)
    return [items[i] for i in items_sort[limit:]]


# def get_connective_candidates(tokens: List[str]):
#     candidates = []
#     sentence = [w.lower().strip("'") for w in tokens]
#     for word_idx, word in enumerate(sentence):
#
#         for conn in distant_connectives:
#             if word == conn[0]:
#                 if all(c in sentence for c in conn[1:]):
#                     candidate = [(word_idx, conn[0])]
#                     try:
#                         i = word_idx
#                         for c in conn[1:]:
#                             i = sentence.index(c, i)
#                             candidate.append((i, c))
#                     except ValueError:
#                         print('distant error...', candidate)
#                         continue
#                     candidates.append(candidate)
#         if word in multi_connectives_first:
#             for multi_conn in multi_connectives:
#                 if (word_idx + len(multi_conn)) <= len(sentence) and all(
#                         c == sentence[word_idx + i] for i, c in enumerate(multi_conn)):
#                     candidates.append([(word_idx + i, c) for i, c in enumerate(multi_conn)])
#         if word in single_connectives:
#             candidates.append([(word_idx, word)])
#     return candidates


def load_pseudo_docs(pseudo_labels_path, filter_empty_paragraphs=False, limit_ratio_pos=0.8, limit_ratio_neg=0.90,
                     paragraph_relation_threshold=0.0, sort_select=True, relation_threshold=0.50, document_limit=0):
    positives = []
    negatives = []
    for line_i, line in enumerate(open(pseudo_labels_path)):
        if 0 < document_limit < line_i:
            break
        par = json.loads(line)
        offset = par['tokens_idx'][0]
        labels = ['O' for _ in par['tokens_idx']]
        probs = []
        par['relations'] = [r for r in par['relations'] if r['is_relation'] > relation_threshold]
        if len(par['relations']) == 0 and not filter_empty_paragraphs:
            par['labels'] = labels
            par['prob'] = np.min(par['probs'])
            if paragraph_relation_threshold > par['prob']:
                continue
            negatives.append(par)
        else:
            for r in par['relations']:
                for t_i, idx in enumerate(r['tokens_idx']):
                    if len(r['tokens_idx']) == 1:
                        label = 'S-ALTLEX'
                    else:
                        label = 'I-ALTLEX'
                    labels[idx - offset] = label
                probs.append(r['is_relation'])
            par['labels'] = labels
            par['prob'] = np.min(probs)
            if paragraph_relation_threshold > par['prob']:
                continue
            positives.append(par)
    # if sort_select:
    #     return sort_select_items(positives, limit_ratio_pos) + sort_select_items(negatives, limit_ratio_neg)
    # else:
    #     items = positives + negatives
    #     limit = int(len(items) * limit_ratio_pos)
    #     return random.sample(items, k=limit)
    return positives + negatives


@click.command()
@click.argument('corpus')
@click.option('--corpus-plus', default="")
@click.option('-b', '--batch-size', type=int, default=8)
@click.option('--split-ratio', type=float, default=0.9)
@click.option('--save-path', default="")
@click.option('--test-set', is_flag=True)
@click.option('--test-seed', default=42, type=int)
@click.option('--valid-seed', default=42, type=int)
@click.option('--weighted-loss', is_flag=True)
@click.option('--continue-model', default="", type=str)
@click.option('--sample-ratio', default=0.0, type=float)
@click.option('-r', '--replace', is_flag=True)
@click.option('--none-weight', default=1.0, type=float)
@click.option('--num-epochs', default=10, type=int)
@click.option('--document-limit', default=0, type=int)
@click.option('--initial-learning-rate', default=1e-4, type=float)
@click.option('--pseudo-limit-ratio', default=0.8, type=float)
@click.option('--paragraph-relation-threshold', default=0.7, type=float)
@click.option('--sort-select', is_flag=True)
@click.option('--val-metric', default='f1-score', type=click.Choice(['f1-score', 'precision', 'recall']))
def main(corpus, corpus_plus, batch_size, split_ratio, save_path, test_set, test_seed, valid_seed, weighted_loss,
         continue_model, sample_ratio, replace, none_weight, num_epochs, document_limit, initial_learning_rate,
         pseudo_limit_ratio, paragraph_relation_threshold, sort_select, val_metric):
    if save_path:
        save_path = Path(save_path)
        if save_path.is_dir() and (save_path / "best_model_altlex_label").exists() and not replace:
            print('LabelModel already exists: Exit without writing.', file=sys.stderr)
            return
    corpus_path = get_corpus_path(corpus)
    train_docs = list(load_docs(corpus_path))
    if test_set:
        train_docs, test_docs = sklearn.model_selection.train_test_split(train_docs, test_size=0.1,
                                                                         random_state=test_seed)
    train_docs, valid_docs = sklearn.model_selection.train_test_split(train_docs, train_size=split_ratio,
                                                                      random_state=valid_seed)

    dataset = train_dataset = ConnDataset(train_docs, filter_empty_paragraphs=sample_ratio > 0.0,
                                          filter_ratio=sample_ratio)
    valid_dataset = ConnDataset(valid_docs, labels=train_dataset.labels)
    print('SAMPLE', len(train_dataset), train_dataset[0])
    print('LABELS:', train_dataset.labels)
    print('Average Sequence Length:', np.mean([len(item['tokens']) for item in dataset.items]))
    print('TRAIN LABEL COUNTS:', train_dataset.get_label_counts())
    print('VALID LABEL COUNTS:', valid_dataset.get_label_counts())

    if corpus_plus:
        pseudo_labels = load_pseudo_docs(corpus_plus,
                                         # limit_ratio_pos=0.40, limit_ratio_neg=0.80,
                                         # sort_select=sort_select,
                                         paragraph_relation_threshold=paragraph_relation_threshold,
                                         relation_threshold=0.33, document_limit=document_limit)
        conn_plus_dataset = ConnDataset([], labels=valid_dataset.labels)
        conn_plus_dataset.add_pseudo_samples(pseudo_labels)
        print('PSEUDO SAMPLE', len(conn_plus_dataset), conn_plus_dataset[0])
        print('PSEUDO LABELS:', conn_plus_dataset.labels)
        print('PSEUDO LABEL COUNTS:', conn_plus_dataset.get_label_counts())
        train_dataset.add_pseudo_samples(pseudo_labels)

    print(f'TRAINING FINAL {len(train_dataset)}')
    print(f'VALIDATION FINAL {len(valid_dataset)}')

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                  collate_fn=ConnDataset.get_collate_fn())
    eval_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                 collate_fn=ConnDataset.get_collate_fn())

    label2id = dataset.labels
    id2label = {v: k for k, v in label2id.items()}

    if continue_model:
        label_save_path = os.path.join(continue_model, f"best_model_altlex_label")
        model = AutoModelForTokenClassification.from_pretrained(label_save_path, local_files_only=True)
        model_save_path = os.path.join(save_path, f"best_model_altlex_label")
        model.save_pretrained(model_save_path)
    else:
        model = AutoModelForTokenClassification.from_pretrained("roberta-base",
                                                                num_labels=dataset.get_num_labels(),
                                                                id2label=id2label, label2id=label2id,
                                                                local_files_only=True,
                                                                hidden_dropout_prob=0.3)
    for param in model.base_model.embeddings.parameters():
        param.requires_grad = False
    for layer in model.base_model.encoder.layer[:10]:
        for param in layer.parameters():
            param.requires_grad = False
    optimizer = AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=0.05)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=10, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    best_score = (0, {})
    epochs_no_improvement = 0

    if weighted_loss:
        label_weights = dataset.get_label_weights()
    else:
        label_weights = [none_weight] + [1.0 for _ in range(dataset.get_num_labels() - 1)]
    print(f"CLASS WEIGHTS: {label_weights}")

    if continue_model:
        metric = evaluate.load("poseval")
        model.eval()
        losses = []
        for batch in tqdm(eval_dataloader, desc='Validation',
                          total=len(eval_dataloader), mininterval=5):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss = compute_loss(dataset.get_num_labels(), label_weights, outputs.logits, batch['labels'],
                                    device)
                losses.append(loss.item())

            preds = torch.argmax(outputs.logits, dim=-1)
            predictions = []
            references = []
            for pred, ref in zip(preds.tolist(), batch['labels'].tolist()):
                pred = [id2label[p] for i, p in enumerate(pred) if ref[i] != -100]
                ref = [id2label[i] for i in ref if i != -100]
                assert len(pred) == len(ref), f"PRED: {pred}, REF {ref}"
                predictions.append(pred)
                references.append(ref)
            metric.add_batch(predictions=predictions, references=references)
        results = metric.compute(zero_division=0)
        print_metrics_results(results)
        print(f"Initial Score: {results['macro avg'][val_metric]}")
        best_score = (results['macro avg'][val_metric], results)

    for epoch in range(num_epochs):
        print(f"##\n## Epoch ({epoch})\n##")
        model.train()
        losses = []

        for batch in tqdm(train_dataloader, desc='Training', total=len(train_dataloader), mininterval=5):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = compute_loss(dataset.get_num_labels(), label_weights, outputs.logits, batch['labels'], device)
            loss.backward()
            losses.append(loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        loss_train = np.mean(losses)

        metric = evaluate.load("poseval")
        model.eval()
        losses = []
        for batch in tqdm(eval_dataloader, desc='Validation',
                          total=len(eval_dataloader), mininterval=5):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss = compute_loss(dataset.get_num_labels(), label_weights, outputs.logits, batch['labels'],
                                    device)
                losses.append(loss.item())

            preds = torch.argmax(outputs.logits, dim=-1)
            predictions = []
            references = []
            for pred, ref in zip(preds.tolist(), batch['labels'].tolist()):
                pred = [id2label[p] for i, p in enumerate(pred) if ref[i] != -100]
                ref = [id2label[i] for i in ref if i != -100]
                assert len(pred) == len(ref), f"PRED: {pred}, REF {ref}"
                predictions.append(pred)
                references.append(ref)
            metric.add_batch(predictions=predictions, references=references)
        loss_valid = np.mean(losses)

        results = metric.compute(zero_division=0)
        print_metrics_results(results)

        print(f'Training loss: {loss_train}')
        print(f'Validation loss: {loss_valid}')
        current_score = (results['macro avg'][val_metric], results)
        if current_score[0] > best_score[0]:
            best_score = current_score
            print(f"Store new best model! Score: {current_score[0]}...")
            if save_path:
                model_save_path = os.path.join(save_path, f"best_model_altlex_label")
                model.save_pretrained(model_save_path)
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= 4:
                print('Early stopping...')
                break
    print_final_results(best_score[0], best_score[1])


if __name__ == '__main__':
    main()
