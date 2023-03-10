import os

import click
import evaluate
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification
from transformers import get_scheduler, AdamW

from helpers.data import get_corpus_path, load_docs
from helpers.labeling import ConnDataset
from helpers.stats import print_metrics_results, print_final_results


def compute_loss(num_labels, weights, logits, labels, device):
    loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    return loss


@click.command()
@click.argument('corpus')
@click.option('-b', '--batch-size', type=int, default=8)
@click.option('--split-ratio', type=float, default=0.8)
@click.option('--save-path', default=".")
@click.option('--test-set', is_flag=True)
@click.option('--weighted-loss', is_flag=True)
@click.option('--random-seed', default=42, type=int)
def main(corpus, batch_size, split_ratio, save_path, test_set, random_seed, weighted_loss):
    corpus_path = get_corpus_path(corpus)
    corpus_docs = load_docs(corpus_path)
    conn_dataset = ConnDataset(corpus_docs, relation_type='altlex')
    print('SAMPLE', len(conn_dataset), conn_dataset[0])
    print('LABELS:', conn_dataset.labels)
    print('LABEL COUNTS:', conn_dataset.get_label_counts())
    train_dataset = conn_dataset
    if test_set:
        dataset_length = len(train_dataset)
        train_size = int(dataset_length * 0.8)
        test_size = dataset_length - train_size
        train_dataset, test_dataset = random_split(conn_dataset, [train_size, test_size],
                                                   generator=torch.Generator().manual_seed(random_seed))

    dataset_length = len(train_dataset)
    train_size = int(dataset_length * split_ratio)
    valid_size = dataset_length - train_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
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
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    num_epochs = 20
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=10, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    best_score = (float('inf'), {})
    epochs_no_improvement = 0

    if weighted_loss:
        label_weights = conn_dataset.get_label_weights()
    else:
        label_weights = [0.5] + [1.0 for _ in range(conn_dataset.get_num_labels() - 1)]
    print(f"CLASS WEIGHTS: {label_weights}")

    for epoch in range(num_epochs):
        model.train()
        losses = []
        for batch_i, batch in tqdm(enumerate(train_dataloader), desc='Training',
                                   total=len(train_dataloader), mininterval=5):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = compute_loss(conn_dataset.get_num_labels(), label_weights, outputs.logits, batch['labels'], device)
            loss.backward()
            losses.append(loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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
                loss = compute_loss(conn_dataset.get_num_labels(), label_weights, outputs.logits, batch['labels'],
                                    device)
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
        print_metrics_results(results)

        print(f'Training loss: {loss_train}')
        print(f'Validation loss: {loss_valid}')
        current_score = (loss_valid, results)
        if current_score[0] < best_score[0]:
            best_score = current_score
            print(f"Store new best model! Score: {current_score[0]}...")
            model_save_path = os.path.join(save_path, f"best_model_altlex_label")
            model.save_pretrained(model_save_path)
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= 3:
                print('Early stopping...')
                break
    print_final_results(best_score[0], best_score[1])


if __name__ == '__main__':
    main()
