import os
from typing import List, Dict
from collections import Counter
from copy import deepcopy

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

from features import features


class CustomDataset(Dataset):
    def __init__(self, data, tag_to_ix, word_to_ix, freq_dict: Dict[str, pd.DataFrame] = None, case_sensitive=True):
        super(CustomDataset, self).__init__()
        self.case_sensitive = case_sensitive
        self.pad = 'PAD'
        self.unk = 'UNK'
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.freq_dict = freq_dict
        self.custom_features = None
        if not case_sensitive:
            self.word_to_ix = {word.lower(): index for word, index in word_to_ix.items()}
            self.pad = self.pad.lower()
            self.unk = self.unk.lower()
        self.data = self.prepare_dataset(deepcopy(data), self.tag_to_ix, self.word_to_ix)
        if self.freq_dict is not None:
            self.custom_features = self.compute_custom_features(deepcopy(data))

    def __getitem__(self, index):
        sentence, tags = self.data[index]
        mask = tags >= 0
        f = (self.custom_features[index] if self.custom_features is not None
             else torch.empty(0))
        return sentence, tags, mask, f

    def __len__(self):
        return len(self.data)

    def prepare_dataset(self, data, tag_to_ix, word_to_ix):
        max_len = self.compute_max_sentence_len(data)
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

    @staticmethod
    def compute_max_sentence_len(data):
        return len(max(data, key=lambda x: len(x[0]))[0])

    def word2features(self, sentence: List[str], i: int, n: int) -> list:
        result = []
        word = sentence[i]

        is_number = features.isnumber(word)
        result.append(is_number)
        probs = features.calculate_probs(
            word, features.getprob_binary, self.freq_dict
        )
        probs = list(probs.values())
        result.extend(probs)

        if i > 0:
            word1 = sentence[i - 1]
            probs = features.calculate_probs(
                f'{word1} {word}', features.getprob_binary, self.freq_dict
            )
            result.extend(list(probs.values()))
        else:
            result.extend([0] * len(self.freq_dict))

        if i < n - 1:
            word1 = sentence[i + 1]
            probs = features.calculate_probs(
                f'{word} {word1}', features.getprob_binary, self.freq_dict
            )
            result.extend(list(probs.values()))
        else:
            result.extend([0] * len(self.freq_dict))

        return result

    def compute_custom_features(self, data) -> List[torch.Tensor]:
        custom_features = []
        max_len = self.compute_max_sentence_len(data)
        for index, (sentence, tags) in enumerate(data):
            n = len(sentence)
            word_features = [self.word2features(sentence, i, n) for i in range(n)]
            pad_features = [[0] * len(word_features[0]) for _ in range(max_len - n)]
            word_features.extend(pad_features)
            custom_features.append(torch.tensor(word_features, dtype=torch.float))
        return custom_features

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
