import glob
import os

import click
import evaluate
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoModelForTokenClassification

from helpers.data import get_corpus_path
from helpers.labeling import ConnDataset


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
def main(corpus, batch_size, save_path, random_seed):
    save_path = os.path.join(save_path, "best_model_altlex_label")
    save_paths = glob.glob(save_path)
    print('SAVE PATHS:', save_paths)
    if len(save_paths) == 0:
        raise ValueError('No models found...')

    corpus_path = get_corpus_path(corpus)
    conn_dataset = ConnDataset(corpus_path, relation_type="altlex")
    dataset_length = len(conn_dataset)
    train_size = int(dataset_length * 0.9)
    test_size = dataset_length - train_size
    _, test_dataset = random_split(conn_dataset, [train_size, test_size],
                                   generator=torch.Generator().manual_seed(random_seed))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if len(save_paths) == 1:
        print("Load Single Classifier Model")
        model = AutoModelForTokenClassification.from_pretrained(save_paths[0])
        id2label = model.config.id2label
        model.eval()
        model.to(device)
        models = [model]
    else:
        print("Load Ensemble Model")
        models = []
        for save_path in save_paths:
            model = AutoModelForTokenClassification.from_pretrained(save_path)
            model.eval()
            model.to(device)
            models.append(model)
        id2label = models[0].config.id2label

    metric = evaluate.load("poseval")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=ConnDataset.get_collate_fn())
    for batch in tqdm(test_dataloader, total=len(test_dataset) // batch_size, mininterval=2):
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = compute_ensemble_prediction(models, batch)
        predictions = []
        references = []
        for pred, ref in zip(preds.tolist(), batch['labels'].tolist()):
            pred = [id2label[p] for i, p in enumerate(pred) if ref[i] != -100]
            ref = [id2label[i] for i in ref if i != -100]
            assert len(pred) == len(ref), f"PRED: {pred}, REF {ref}"
            predictions.append(pred)
            references.append(ref)
        metric.add_batch(predictions=predictions, references=references)

    results = metric.compute()
    for key, vals in results.items():
        if key == 'accuracy':
            print(f"{key:10}  {vals * 100:02.2f}")
        else:
            print(
                f"{key:10}  {vals['precision'] * 100:02.2f}  {vals['recall'] * 100:02.2f}  {vals['f1-score'] * 100:02.2f}  {vals['support']}")


if __name__ == '__main__':
    main()
