import os
from typing import List
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from copy import deepcopy


class CustomDataset(Dataset):
    def __init__(self, data, tag_to_ix, word_to_ix, case_sensitive=True):
        super(CustomDataset, self).__init__()
        self.case_sensitive = case_sensitive
        self.pad = 'PAD'
        self.unk = 'UNK'
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        if not case_sensitive:
            self.word_to_ix = {word.lower(): index for word, index in word_to_ix.items()}
            self.pad = self.pad.lower()
            self.unk = self.unk.lower()
        self.data = self.prepare_dataset(deepcopy(data), self.tag_to_ix, self.word_to_ix)

    def __getitem__(self, index):
        sentence, tags = self.data[index]
        mask = tags >= 0
        return sentence, tags, mask

    def __len__(self):
        return len(self.data)

    def prepare_dataset(self, data, tag_to_ix, word_to_ix):
        max_len = len(max(data, key=lambda x: len(x[0]))[0])
        for index, (sentence, tags) in enumerate(data):
            if not self.case_sensitive:
                sentence = [word.lower() for word in sentence]
            sentence = [word_to_ix[word] if word in word_to_ix else word_to_ix[self.unk] for word in sentence]
            tags = [tag_to_ix[tag] for tag in tags]

            sentence.extend([word_to_ix[self.pad]] * (max_len - len(sentence)))
            tags.extend([-1] * (max_len - len(tags)))

            data[index] = (
                torch.LongTensor(sentence),
                torch.LongTensor(tags)
            )
        return data
    
    def count(self, word: str):
        if not self.case_sensitive:
            word = word.lower()
        if word not in self.word_to_ix:
            raise ValueError(f'Tag: {word} is not in word_to_ix dictionary keys')
        word_ix = self.word_to_ix[word]
        count_word = 0
        for sentence, _ in self.data:
            count_word += torch.sum(sentence == word_ix).item()
        return count_word


def train(model, optimizer, dataloaders, device, num_epochs, output_dir, scheduler=None, tqdm=None, verbose=True):
    losses = {'train': [], 'val': []}
    best_loss = None

    for epoch in range(1, num_epochs + 1) if tqdm is None else tqdm(range(1, num_epochs + 1)):
        losses_per_epoch = {'train': 0.0, 'val': 0.0}

        model.train()
        for x_batch, y_batch, mask_batch in dataloaders['train']:
            x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(x_batch, y_batch, mask_batch)
            loss.backward()
            optimizer.step()
            losses_per_epoch['train'] += loss.item()

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch, mask_batch in dataloaders['val']:
                x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)
                loss = model.neg_log_likelihood(x_batch, y_batch, mask_batch)
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


def get_model_confidence(model: nn.Module, X_test: List[torch.Tensor], device) -> List[float]:
    """Computes model's confidence for each sentence in X_test"""
    confs = []
    with torch.no_grad():
        for sentence in X_test:
            sentence = sentence.unsqueeze(0).to(device)
            best_tag_sequence = model(sentence)
            confidence = torch.exp(
                -model.neg_log_likelihood(sentence, torch.tensor(best_tag_sequence, device=device))
            )
            confs.append(confidence.item())
    return confs
