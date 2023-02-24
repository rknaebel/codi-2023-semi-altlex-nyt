import glob

import click
import evaluate
import torch
from torch.utils.data import DataLoader, random_split

from helpers.data import get_corpus_path
from helpers.senses import ConnSenseDataset, DiscourseSenseClassifier, DiscourseSenseEnsembleClassifier


def load_dataset(corpus, bert_model, relation_type, labels_coarse=None, labels_fine=None, random_seed=42):
    corpus_path = get_corpus_path(corpus)

    conn_dataset = ConnSenseDataset(corpus_path, bert_model, relation_type=relation_type,
                                    labels_coarse=labels_coarse, labels_fine=labels_fine)
    train_dataset = conn_dataset
    dataset_length = len(train_dataset)
    train_size = int(dataset_length * 0.9)
    test_size = dataset_length - train_size
    train_dataset, test_dataset = random_split(conn_dataset, [train_size, test_size],
                                               generator=torch.Generator().manual_seed(random_seed))
    return conn_dataset, test_dataset


@click.command()
@click.argument('corpus')
@click.option('-b', '--batch-size', type=int, default=8)
@click.option('--bert-model', default="roberta-base")
@click.option('--save-path', default=".")
@click.option('--random-seed', default=42, type=int)
def main(corpus, batch_size, bert_model, save_path, random_seed):
    relation_type = 'altlex'
    dataset, altlex_test_dataset = load_dataset(corpus, bert_model, 'altlex', random_seed=random_seed)

    eval_altlex_dataloader = DataLoader(altlex_test_dataset, batch_size=batch_size,
                                        collate_fn=ConnSenseDataset.get_collate_fn())

    id2label_coarse = {v: k for k, v in dataset.labels_coarse.items()}
    id2label_fine = {v: k for k, v in dataset.labels_fine.items()}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    save_paths = glob.glob(save_path)
    if len(save_paths) == 1:
        print("Load Single Classifier Model")
        model = DiscourseSenseClassifier.load(save_path, relation_type=relation_type)
        model.to(device)
    else:
        print("Load Ensemble Model")
        model = DiscourseSenseEnsembleClassifier.load(save_paths, device, relation_type=relation_type)

    metric_coarse = evaluate.load("poseval")
    metric_fine = evaluate.load("poseval")
    model.eval()
    scores = []
    print(f"##\n## EVAL \n##")
    for batch in eval_altlex_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model.predict(batch['inputs'])
        references = [id2label_coarse[i] for i in batch['labels_coarse'].tolist()]
        metric_coarse.add_batch(predictions=[output['coarse']], references=[references])
        references = [id2label_fine[i] for i in batch['labels_fine'].tolist()]
        metric_fine.add_batch(predictions=[output['fine']], references=[references])

    results = metric_coarse.compute(zero_division=0)
    for key, vals in results.items():
        if key == 'accuracy':
            print(f"{key:32}  {vals * 100:02.2f}")
        else:
            print(
                f"{key:32}  {vals['precision'] * 100:02.2f}  {vals['recall'] * 100:02.2f}  {vals['f1-score'] * 100:02.2f}  {vals['support']}")
    scores.append(results['macro avg']['f1-score'])
    print('## ' + '= ' * 50)

    results = metric_fine.compute(zero_division=0)
    for key, vals in results.items():
        if key == 'accuracy':
            print(f"{key:32}  {vals * 100:02.2f}")
        else:
            print(
                f"{key:32}  {vals['precision'] * 100:02.2f}  {vals['recall'] * 100:02.2f}  {vals['f1-score'] * 100:02.2f}  {vals['support']}")
    scores.append(results['macro avg']['f1-score'])


if __name__ == '__main__':
    main()
