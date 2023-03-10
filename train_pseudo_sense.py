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


@click.command()
@click.argument('corpus')
@click.argument('predictions')
@click.option('-b', '--batch-size', type=int, default=8)
@click.option('--split-ratio', type=float, default=0.8)
@click.option('--bert-model', default="roberta-base")
@click.option('--save-path', default=".")
@click.option('--test-set', is_flag=True)
@click.option('--random-seed', default=42, type=int)
def main(corpus, predictions, batch_size, split_ratio, bert_model, save_path, test_set, random_seed):
    dataset, train_dataset, valid_dataset, _ = load_dataset(corpus, bert_model, 'altlex', split_ratio,
                                                            test_set=test_set, random_seed=random_seed,
                                                            predictions=predictions)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                  collate_fn=ConnSenseDataset.get_collate_fn())
    eval_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                 collate_fn=ConnSenseDataset.get_collate_fn())

    model = DiscourseSenseClassifier(len(train_dataset[0]['input']),
                                     dataset.labels_coarse, dataset.labels_fine,
                                     relation_type='altlex')
    id2label_coarse = {v: k for k, v in model.label2id_coarse.items()}
    id2label_fine = {v: k for k, v in model.label2id_fine.items()}

    optimizer = AdamW(model.parameters(), lr=5e-5, amsgrad=True)
    ce_loss_fn = nn.CrossEntropyLoss()

    num_epochs = 20
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=50, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    best_score = (float('inf'), {})
    epochs_no_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        losses = []
        for batch_i, batch in tqdm(enumerate(train_dataloader), desc='Training',
                                   total=len(train_dataloader), mininterval=5):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits_coarse, logits_fine = model(batch['inputs'])
            loss_coarse = ce_loss_fn(logits_coarse, batch['labels_coarse'])
            loss_fine = ce_loss_fn(logits_fine, batch['labels_fine'])
            loss = (loss_coarse + loss_fine) / 2
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        loss_train = np.mean(losses)

        metric_coarse = evaluate.load("poseval")
        metric_fine = evaluate.load("poseval")
        model.eval()
        losses = []
        print(f"##\n## EVAL ({epoch})\n##")
        for batch in tqdm(eval_dataloader, desc='Validation',
                          total=len(eval_dataloader), mininterval=5):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model.predict(batch['inputs'])
            loss_coarse = ce_loss_fn(output['coarse_logits'], batch['labels_coarse'])
            loss_fine = ce_loss_fn(output['fine_logits'], batch['labels_fine'])
            loss = (loss_coarse + loss_fine) / 2
            losses.append(loss.item())

            references = [id2label_coarse[i] for i in batch['labels_coarse'].tolist()]
            metric_coarse.add_batch(predictions=[output['coarse']], references=[references])
            references = [id2label_fine[i] for i in batch['labels_fine'].tolist()]
            metric_fine.add_batch(predictions=[output['fine']], references=[references])
        loss_valid = np.mean(losses)

        results_coarse = metric_coarse.compute(zero_division=0)
        print_metrics_results(results_coarse)

        results_fine = metric_fine.compute(zero_division=0)
        print_metrics_results(results_fine)

        print(f'Training loss: {loss_train}')
        print(f'Validation loss: {loss_valid}')
        current_score = (loss_valid, results_coarse)
        if current_score[0] < best_score[0]:
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
            model.save(save_path, model_state)
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= 3:
                print('Early stopping...')
                break
    print_final_results(best_score[0], best_score[1])


if __name__ == '__main__':
    main()
