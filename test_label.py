import glob
import itertools
import os
from collections import defaultdict

import click
import evaluate
import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification

from helpers.data import get_corpus_path, load_docs
from helpers.evaluate import score_paragraphs
from helpers.labeling import ConnDataset, decode_labels


def compute_ensemble_prediction(models, batch):
    predictions = []
    with torch.no_grad():
        for model in models:
            outputs = model(**batch)
            predictions.append(F.softmax(outputs.logits, dim=-1))
    return torch.argmax(torch.sum(torch.stack(predictions), dim=0), dim=-1)


@click.command()
@click.argument('corpus')
@click.option('-b', '--batch-size', type=int, default=8)
@click.option('--save-path', default=".")
@click.option('--random-seed', default=42, type=int)
@click.option('--mode', default="average", type=click.Choice(['average', 'ensemble']))
def main(corpus, batch_size, save_path, random_seed, mode):
    save_path = os.path.join(save_path, "best_model_altlex_label")
    save_paths = sorted(glob.glob(save_path))
    print('SAVE PATHS:', list(enumerate(save_paths)))
    if len(save_paths) == 0:
        raise ValueError('No models found...')

    corpus_path = get_corpus_path(corpus)
    train_docs = list(load_docs(corpus_path))
    _, test_docs = sklearn.model_selection.train_test_split(train_docs, test_size=0.2,
                                                            random_state=random_seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("Load Model(s)")
    models = []
    for save_path in save_paths:
        model = AutoModelForTokenClassification.from_pretrained(save_path, local_files_only=True)
        model.eval()
        models.append(model)
        print(f'-- loaded {save_path}')
    id2label = models[0].config.id2label
    label2id = {v: k for k, v in id2label.items()}

    def combinations(models, limit=4):
        yield from itertools.combinations(enumerate(models), 1)
        for i in range(3, limit + 1):
            yield from itertools.combinations(enumerate(models), i)

    ensemble_size = 1 if mode == 'average' else 3
    results_all = defaultdict(list)
    test_dataset = ConnDataset(test_docs, labels=label2id)

    for models_id_select in combinations(models, ensemble_size):
        signals_pred = []
        signals_gold = []

        model_ids = [i for i, m in models_id_select]
        models_select = [m for i, m in models_id_select]
        print(f"\n=== Evaluate models: {model_ids}")
        for m in models_select:
            m.to(device)
        metric = evaluate.load("poseval")
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=ConnDataset.get_collate_fn())
        for batch in tqdm(test_dataloader, total=len(test_dataset) // batch_size, mininterval=5):
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = compute_ensemble_prediction(models_select, batch)
            predictions = []
            references = []
            for pred, ref in zip(preds.tolist(), batch['labels'].tolist()):
                pred = [id2label[p] for i, p in enumerate(pred) if ref[i] != -100]
                ref = [id2label[i] for i in ref if i != -100]
                assert len(pred) == len(ref), f"PRED: {pred}, REF {ref}"
                predictions.append(pred)
                references.append(ref)
                signals_pred.append([[i for p, i in signal] for signal in decode_labels(pred, pred)])
                signals_gold.append([[i for p, i in signal] for signal in decode_labels(ref, ref)])
            metric.add_batch(predictions=predictions, references=references)
        for m in models_select:
            m.to('cpu')
        results = metric.compute()

        precision, recall, f1 = score_paragraphs(signals_gold, signals_pred)
        results['Full-strong'] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': 0,
        }
        precision, recall, f1 = score_paragraphs(signals_gold, signals_pred, threshold=0.7)
        results['Full-soft'] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': 0,
        }

        for key, vals in results.items():
            if key == 'accuracy':
                # print(f"{key:10}  {vals * 100:02.2f}")
                pass
            else:
                print(f"{key:15} "
                      f"{vals['precision'] * 100:05.2f}  {vals['recall'] * 100:05.2f}  {vals['f1-score'] * 100:05.2f}  "
                      f"{vals['support']}")
                results_all[key].append((vals['precision'], vals['recall'], vals['f1-score']))

    if mode == 'average':
        print('== Final MEAN Performance')
        for key, vals in results_all.items():
            vals = np.stack(vals)
            vals_mean = vals.mean(axis=0)
            vals_var = vals.std(axis=0)
            print(f"{key:15} "
                  f"{vals_mean[0] * 100: 5.2f} ({vals_var[0] * 100: 5.2f})  "
                  f"{vals_mean[1] * 100: 5.2f} ({vals_var[1] * 100: 5.2f})  "
                  f"{vals_mean[2] * 100: 5.2f} ({vals_var[2] * 100: 5.2f})")


if __name__ == '__main__':
    main()
