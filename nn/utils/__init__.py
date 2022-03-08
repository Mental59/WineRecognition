import os
from typing import List
import torch
from torch import nn
from matplotlib import pyplot as plt
from .custom_dataset import CustomDataset


def train(model, optimizer, dataloaders, device, num_epochs, output_dir, scheduler=None, tqdm=None, verbose=True):
    losses = {'train': [], 'val': []}
    best_loss = None

    for epoch in range(1, num_epochs + 1) if tqdm is None else tqdm(range(1, num_epochs + 1)):
        losses_per_epoch = {'train': 0.0, 'val': 0.0}

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
            losses_per_epoch['train'] += loss.item()

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch, mask_batch, custom_features in dataloaders['val']:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                custom_features = custom_features.to(device)
                loss = model.neg_log_likelihood(x_batch, y_batch, mask_batch, custom_features)
                losses_per_epoch['val'] += loss.item()

        for mode in ['train', 'val']:
            losses_per_epoch[mode] /= len(dataloaders[mode])
            losses[mode].append(losses_per_epoch[mode])

        if best_loss is None or best_loss > losses_per_epoch['val']:
            best_loss = losses_per_epoch['val']
            torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))

        if scheduler is not None:
            scheduler.step(losses_per_epoch['val'])

        if verbose:
            print(
                'Epoch: {}'.format(epoch),
                'train_loss: {}'.format(losses_per_epoch['train']),
                'val_loss: {}'.format(losses_per_epoch['val']),
                sep=', '
            )

    model.load_state_dict(torch.load(os.path.join(output_dir, 'model.pth')))
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


def get_model_confidence(model: nn.Module, X_test: List[torch.Tensor], device, custom_features=None) -> List[float]:
    """Computes model's confidence for each sentence in X_test"""
    confs = []
    with torch.no_grad():
        for index, sentence in enumerate(X_test):
            sentence = sentence.unsqueeze(0).to(device)
            f = (custom_features[index][:sentence.size(1), ...].unsqueeze(0).to(device)
                 if custom_features is not None else None)
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
