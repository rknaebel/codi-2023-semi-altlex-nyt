import sys
from pathlib import Path

import click
import evaluate
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from helpers.senses import ConnSenseDataset, DiscourseSenseClassifier, load_dataset
from helpers.stats import print_metrics_results, print_final_results


def compute_loss(num_labels, weights, logits, labels, device):
    loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device), label_smoothing=0.1)
    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    return loss


@click.command()
@click.argument('corpus')
@click.argument('predictions')
@click.option('-b', '--batch-size', type=int, default=8)
@click.option('--split-ratio', type=float, default=0.8)
@click.option('--bert-model', default="roberta-base")
@click.option('--save-path', default="")
@click.option('--test-set', is_flag=True)
@click.option('--random-seed', default=42, type=int)
@click.option('--hidden', default="2048,512")
@click.option('--drop-rate', default=0.4, type=float)
@click.option('-r', '--replace', is_flag=True)
def main(corpus, predictions, batch_size, split_ratio, bert_model, save_path, test_set, random_seed, hidden, drop_rate,
         replace):
    save_path = Path(save_path)
    if save_path.is_dir() and (save_path / "best_model_altlex_sense.pt").exists() and not replace:
        print('SenseModel already exists: Exit without writing.', file=sys.stderr)
        return

    dataset, train_dataset, valid_dataset, _ = load_dataset(corpus, bert_model, 'altlex', split_ratio,
                                                            labels_coarse={'None': 0}, labels_fine={'None': 0},
                                                            test_set=test_set, random_seed=random_seed,
                                                            predictions=predictions)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                  collate_fn=ConnSenseDataset.get_collate_fn())
    eval_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                 collate_fn=ConnSenseDataset.get_collate_fn())

    model = DiscourseSenseClassifier(len(train_dataset[0]['input']),
                                     dataset.labels_coarse, dataset.labels_fine,
                                     relation_type='altlex',
                                     hidden=[int(i) for i in hidden.split(',')],
                                     drop_rate=drop_rate)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    num_epochs = 20
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=10, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    best_score = (0, {})
    epochs_no_improvement = 0
    weights_coarse = [0.8] + [1.0] * (dataset.get_num_labels_coarse() - 1)
    weights_fine = [0.8] + [1.0] * (dataset.get_num_labels_fine() - 1)

    for epoch in range(num_epochs):
        model.train()
        losses = []
        for batch_i, batch in tqdm(enumerate(train_dataloader), desc='Training',
                                   total=len(train_dataloader), mininterval=5):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits_coarse, logits_fine = model(batch['inputs'])
            loss_coarse = compute_loss(dataset.get_num_labels_coarse(), weights_coarse,
                                       logits_coarse, batch['labels_coarse'], device)
            loss_fine = compute_loss(dataset.get_num_labels_fine(), weights_fine,
                                     logits_fine, batch['labels_fine'], device)
            loss = (loss_coarse * 3 + loss_fine) / 4
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        loss_train = np.mean(losses)

        metric_coarse = evaluate.load("poseval")
        model.eval()
        losses = []
        print(f"##\n## EVAL ({epoch})\n##")
        for batch in tqdm(eval_dataloader, desc='Validation',
                          total=len(eval_dataloader), mininterval=5):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model.predict(batch['inputs'])
            loss_coarse = compute_loss(dataset.get_num_labels_coarse(), weights_coarse,
                                       output['coarse_logits'], batch['labels_coarse'], device)
            loss = loss_coarse
            losses.append(loss.item())
            references = [model.id2label_coarse[i] for i in batch['labels_coarse'].tolist()]
            metric_coarse.add_batch(predictions=[output['coarse']], references=[references])
        loss_valid = np.mean(losses)

        results_coarse = metric_coarse.compute(zero_division=0)
        print_metrics_results(results_coarse)

        print(f'Training loss: {loss_train}')
        print(f'Validation loss: {loss_valid}')
        current_score = (results_coarse['None']['f1-score'], results_coarse)
        if current_score[0] > best_score[0]:
            best_score = current_score
            print(f"Store new best model! Score: {current_score[0]}...")
            model_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "sched": lr_scheduler.state_dict(),
                "score": current_score,
                "config": model.config,
            }
            if save_path:
                model.save(save_path, model_state)
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= 4:
                print('Early stopping...')
                break
    print_final_results(best_score[0], best_score[1])


if __name__ == '__main__':
    main()
