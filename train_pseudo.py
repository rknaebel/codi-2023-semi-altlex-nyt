import os

import click
import evaluate
import numpy as np
import pandas as pd
import torch
from discopy_data.data.doc import Relation
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification
from transformers import get_scheduler

import helpers
from helpers.data import get_corpus_path, load_docs
from helpers.labeling import ConnDataset


def compute_loss(num_labels, logits, labels, device):
    weights = [0.75] + ([1.25] * (num_labels - 1))
    loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    return loss


def load_pseudo_docs(pseudo_labels_path):
    column_names = helpers.stats.column_names
    labels = pd.read_csv(pseudo_labels_path, names=column_names)
    docs = {}
    for corpus in labels.corpus.unique():
        for doc in load_docs(get_corpus_path(corpus)):
            docs[doc.doc_id] = doc
    for (corpus, doc_id), label_group in labels.groupby(['corpus', 'doc_id']):
        doc = docs[doc_id]
        words = doc.get_tokens()
        relations = label_group.to_dict(orient='records')
        relations = [
            Relation([], [], [words[i] for i in map(int, rel['indices'].split('-'))], [rel['sense2']], rel['type'])
            for rel in relations
        ]
        yield doc.with_relations(relations)


@click.command()
@click.argument('corpus')
@click.argument('corpus-plus-path')
@click.option('-b', '--batch-size', type=int, default=8)
@click.option('--split-ratio', type=float, default=0.8)
@click.option('--save-path', default=".")
@click.option('--test-set', is_flag=True)
@click.option('--random-seed', default=42, type=int)
def main(corpus, corpus_plus_path, batch_size, split_ratio, save_path, test_set, random_seed):
    if os.path.exists(save_path):
        raise FileExistsError(f"STOP: Model {save_path} already exists!")

    corpus_path = get_corpus_path(corpus)
    corpus_docs = load_docs(corpus_path)
    conn_dataset = ConnDataset(corpus_docs, relation_type='altlex')
    print('SAMPLE', len(conn_dataset), conn_dataset[0])
    print('LABELS:', conn_dataset.labels)
    print('LABEL COUNTS:', conn_dataset.get_label_counts())
    train_dataset = conn_dataset
    if test_set:
        dataset_length = len(train_dataset)
        train_size = int(dataset_length * 0.9)
        test_size = dataset_length - train_size
        train_dataset, test_dataset = random_split(conn_dataset, [train_size, test_size],
                                                   generator=torch.Generator().manual_seed(random_seed))

    dataset_length = len(train_dataset)
    train_size = int(dataset_length * split_ratio)
    valid_size = dataset_length - train_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    corpus_docs = load_pseudo_docs(corpus_plus_path)
    conn_plus_dataset = ConnDataset(corpus_docs, relation_type='altlex', filter_empty_paragraphs=True,
                                    labels=conn_dataset.labels)
    print('PSEUDO SAMPLE', len(conn_plus_dataset), conn_plus_dataset[0])
    print('PSEUDO LABELS:', conn_plus_dataset.labels)
    print('PSEUDO LABEL COUNTS:', conn_plus_dataset.get_label_counts())

    train_dataset = ConcatDataset([train_dataset, conn_plus_dataset])

    print(len(train_dataset), len(valid_dataset))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                  collate_fn=ConnDataset.get_collate_fn())
    eval_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                 collate_fn=ConnDataset.get_collate_fn())

    label2id = conn_dataset.labels
    id2label = {v: k for k, v in label2id.items()}

    model = AutoModelForTokenClassification.from_pretrained("roberta-base",
                                                            num_labels=conn_dataset.get_num_labels(),
                                                            id2label=id2label, label2id=label2id)
    for param in model.base_model.embeddings.parameters():
        param.requires_grad = False
    for layer in model.base_model.encoder.layer[:6]:
        for param in layer.parameters():
            param.requires_grad = False
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 50
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=50, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps), mininterval=3)

    best_score = 0.0
    epochs_no_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        losses = []
        for batch_i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = compute_loss(conn_dataset.get_num_labels(), outputs.logits, batch['labels'], device)
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        loss_train = np.mean(losses)

        metric = evaluate.load("poseval")
        model.eval()
        losses = []
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss = compute_loss(conn_dataset.get_num_labels(), outputs.logits, batch['labels'], device)
                losses.append(loss.item())

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
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

        results = metric.compute()
        for key, vals in results.items():
            if key == 'accuracy':
                print(f"{key:10}  {vals * 100:02.2f}")
            else:
                print(
                    f"{key:10}  {vals['precision'] * 100:02.2f}  {vals['recall'] * 100:02.2f}  {vals['f1-score'] * 100:02.2f}  {vals['support']}")

        print(f'Training loss: {loss_train}')
        print(f'Validation loss: {loss_valid}')
        current_score = results['macro avg']['f1-score']
        if current_score > best_score:
            best_score = current_score
            print(f"Store new best model! Score: {current_score}...")
            model_save_path = os.path.join(save_path, f"best_model_altlex_label")
            model.save_pretrained(model_save_path)
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement > 3:
                print('Early stopping...')
                break


if __name__ == '__main__':
    main()
