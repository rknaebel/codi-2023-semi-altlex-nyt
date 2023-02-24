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


@click.command()
@click.argument('corpus')
@click.argument('relation-type')
@click.option('-b', '--batch-size', type=int, default=8)
@click.option('--split-ratio', type=float, default=0.8)
@click.option('--bert-model', default="roberta-base")
@click.option('--save-path', default=".")
@click.option('--test-set', is_flag=True)
@click.option('--random-seed', default=42, type=int)
def main(corpus, relation_type, batch_size, split_ratio, bert_model, save_path, test_set, random_seed):
    dataset, train_dataset, valid_dataset, _ = load_dataset(corpus, bert_model, relation_type, split_ratio,
                                                            test_set=test_set, random_seed=random_seed)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                  collate_fn=ConnSenseDataset.get_collate_fn())
    eval_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                 collate_fn=ConnSenseDataset.get_collate_fn())

    model = DiscourseSenseClassifier(len(train_dataset[0]['input']),
                                     dataset.labels_coarse, dataset.labels_fine,
                                     relation_type=relation_type)
    id2label_coarse = {v: k for k, v in model.label2id_coarse.items()}
    id2label_fine = {v: k for k, v in model.label2id_fine.items()}

    optimizer = AdamW(model.parameters(), lr=5e-4, amsgrad=True)
    ce_loss_fn = nn.CrossEntropyLoss()

    num_epochs = 50
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=50, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    best_score = 0.0
    epochs_no_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        losses = []
        for batch_i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits_coarse, logits_fine = model(batch['inputs'])
            loss_coarse = ce_loss_fn(logits_coarse, batch['labels_coarse'])
            loss_fine = ce_loss_fn(logits_fine, batch['labels_fine'])
            loss = loss_coarse + loss_fine
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        loss_train = np.mean(losses)

        metric_coarse = evaluate.load("poseval")
        metric_fine = evaluate.load("poseval")
        model.eval()
        scores = []
        losses = []
        print(f"##\n## EVAL ({epoch}) {relation_type}\n##")
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model.predict(batch['inputs'])
            loss_coarse = ce_loss_fn(output['coarse_logits'], batch['labels_coarse'])
            loss_fine = ce_loss_fn(output['fine_logits'], batch['labels_fine'])
            loss = loss_coarse + loss_fine
            losses.append(loss.item())

            references = [id2label_coarse[i] for i in batch['labels_coarse'].tolist()]
            metric_coarse.add_batch(predictions=[output['coarse']], references=[references])
            references = [id2label_fine[i] for i in batch['labels_fine'].tolist()]
            metric_fine.add_batch(predictions=[output['fine']], references=[references])
        loss_valid = np.mean(losses)

        results = metric_coarse.compute()
        for key, vals in results.items():
            if key == 'accuracy':
                print(f"{key:32}  {vals * 100:02.2f}")
            else:
                print(
                    f"{key:32}  {vals['precision'] * 100:02.2f}  {vals['recall'] * 100:02.2f}  {vals['f1-score'] * 100:02.2f}  {vals['support']}")
        scores.append(results['macro avg']['f1-score'])
        print('## ' + '= ' * 50)

        results = metric_fine.compute()
        for key, vals in results.items():
            if key == 'accuracy':
                print(f"{key:32}  {vals * 100:02.2f}")
            else:
                print(
                    f"{key:32}  {vals['precision'] * 100:02.2f}  {vals['recall'] * 100:02.2f}  {vals['f1-score'] * 100:02.2f}  {vals['support']}")
        scores.append(results['macro avg']['f1-score'])
        print('## ' + '= ' * 50)

        print(f'Training loss:   {loss_train}')
        print(f'Validation loss: {loss_valid}')
        current_score = np.mean(scores)
        if current_score > best_score:
            print(f"Store new best model! Score: {current_score}...")
            best_score = current_score
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
            if epochs_no_improvement > 3:
                print('Early stopping...')
                break


if __name__ == '__main__':
    main()
