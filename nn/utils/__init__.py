import os
from typing import List
import torch
import wandb
from torch import nn
from matplotlib import pyplot as plt
from .custom_dataset import CustomDataset


def train(
        model,
        optimizer,
        dataloaders,
        device,
        num_epochs,
        output_dir,
        neptune_run=None,
        log_wandb=False,
        scheduler=None,
        tqdm=None,
        verbose=True):
    losses = {'train': [], 'val': []}
    best_loss = None
    model_path = os.path.join(output_dir, 'model.pth')

    for epoch in range(1, num_epochs + 1) if tqdm is None else tqdm(range(1, num_epochs + 1)):
        losses_per_epoch = {'train': 0.0, 'val': 0.0}

        if neptune_run is not None:
            neptune_run['epoch'] = epoch
        if log_wandb:
            wandb.log({'epoch': epoch})

        model.train()
        for x_batch, y_batch, mask_batch, custom_features in dataloaders['train']:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            custom_features = custom_features.to(device)
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(x_batch, y_batch, mask_batch, custom_features)
            loss.backward()
            optimizer.step()

            if neptune_run is not None:
                neptune_run['train/batch/loss'].append(loss.item())
            if log_wandb:
                wandb.log({'train': {'batch': {'loss': loss.item()}}})

            losses_per_epoch['train'] += loss.item()

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch, mask_batch, custom_features in dataloaders['val']:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                custom_features = custom_features.to(device)
                loss = model.neg_log_likelihood(x_batch, y_batch, mask_batch, custom_features)

                if neptune_run is not None:
                    neptune_run['val/batch/loss'].append(loss.item())
                if log_wandb:
                    wandb.log({'val': {'batch': {'loss': loss.item()}}})

                losses_per_epoch['val'] += loss.item()

        for mode in ['train', 'val']:
            losses_per_epoch[mode] /= len(dataloaders[mode])
            losses[mode].append(losses_per_epoch[mode])

        if best_loss is None or best_loss > losses_per_epoch['val']:
            best_loss = losses_per_epoch['val']
            torch.save(model.state_dict(), model_path)

        if scheduler is not None:
            scheduler.step(losses_per_epoch['val'])

        if verbose:
            print(
                'Epoch: {}'.format(epoch),
                'train_loss: {}'.format(losses_per_epoch['train']),
                'val_loss: {}'.format(losses_per_epoch['val']),
                sep=', '
            )

    if os.path.exists(model_path):
        if neptune_run is not None:
            neptune_run['model_checkpoints/best_model'].upload(model_path)
        if log_wandb:
            model_artifact = wandb.Artifact('best_model', type='model')
            model_artifact.add_file(model_path, 'model.pth')
            wandb.log_artifact(model_artifact)

        model.load_state_dict(torch.load(model_path))

    return model, losses


def plot_losses(losses, figsize=(12, 8), savepath: str = None, show=True):
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=figsize)
    for mode in ['train', 'val']:
        plt.plot(losses[mode], label=mode)
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()


def generate_tag_to_ix(keys: list):
    tag_to_ix = {}
    i = 0
    for key in keys:
        tag_to_ix[key] = i
        i += 1
    return tag_to_ix


def get_model_confidence(
        model: nn.Module, X_test: List[torch.Tensor], device, test_dataset: CustomDataset = None) -> List[float]:
    """Computes model's confidence for each sentence in X_test"""
    confs = []
    with torch.no_grad():
        for index, sentence in enumerate(X_test):
            sentence = sentence.unsqueeze(0).to(device)

            f = None
            if test_dataset is not None:
                _, _, _, custom_features = test_dataset[index]
                if custom_features is not None:
                    f = custom_features[:sentence.size(1), ...].unsqueeze(0).to(device)

            best_tag_sequence = model(sentence, custom_features=f)
            confidence = torch.exp(
                -model.neg_log_likelihood(
                    sentence,
                    torch.tensor(best_tag_sequence, device=device),
                    custom_features=f
                )
            )
            confs.append(confidence.item())

    return confs

def get_model_mean_confidence(
        model: nn.Module,
        X_test: List[torch.Tensor],
        device,
        tqdm,
        test_dataset: CustomDataset = None) -> float:
    """Computes model's confidence for each sentence in X_test"""
    conf = 0
    with torch.no_grad():
        for index, sentence in tqdm(enumerate(X_test)):
            sentence = sentence.unsqueeze(0).to(device)

            f = None
            if test_dataset is not None:
                _, _, _, custom_features = test_dataset[index]
                if custom_features is not None:
                    f = custom_features[:sentence.size(1), ...].unsqueeze(0).to(device)

            best_tag_sequence = model(sentence, custom_features=f)
            confidence = torch.exp(
                -model.neg_log_likelihood(
                    sentence,
                    torch.tensor(best_tag_sequence, device=device),
                    custom_features=f
                )
            )
            conf += confidence.item()

    return conf / len(X_test)
