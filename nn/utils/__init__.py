import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from copy import deepcopy


class CustomDataset(Dataset):
    def __init__(self, data, tag_to_ix, word_to_ix):
        super(CustomDataset, self).__init__()
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.data = self.prepare_dataset(deepcopy(data), tag_to_ix, word_to_ix)

    @staticmethod
    def prepare_dataset(data, tag_to_ix, word_to_ix):
        max_len = len(max(data, key=lambda x: len(x[0]))[0])
        for index, (sentence, tags) in enumerate(data):
            sentence = [word_to_ix[word] if word in word_to_ix else word_to_ix['UNK'] for word in sentence]
            tags = [tag_to_ix[tag] for tag in tags]

            sentence.extend([word_to_ix['PAD']] * (max_len - len(sentence)))
            tags.extend([-1] * (max_len - len(tags)))

            data[index] = (
                torch.LongTensor(sentence),
                torch.LongTensor(tags)
            )

        return data
    
    def count(self, word: str):
        if word not in self.word_to_ix:
            raise ValueError(f'Tag: {word} is not in word_to_ix dictionary keys')
        word_ix = self.word_to_ix[word]
        count_word = 0
        for sentence, _ in self.data:
            count_word += torch.sum(sentence == word_ix).item()
        return count_word


    def __getitem__(self, index):
        sentence, tags = self.data[index]
        mask = tags >= 0
        return sentence, tags, mask

    def __len__(self):
        return len(self.data)


def train(model, optimizer, dataloaders, device, num_epochs, tqdm=None, verbose=True):
    losses = {'train': [], 'val': []}
    best_model_wts = None
    best_loss = None

    for epoch in range(1, num_epochs + 1) if tqdm is None else tqdm(range(1, num_epochs + 1)):
        losses_per_batch = {'train': 0.0, 'val': 0.0}

        model.train()
        for x_batch, y_batch, mask_batch in dataloaders['train']:
            x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)
            optimizer.zero_grad()
            loss = model.neg_log_likehood(x_batch, y_batch, mask_batch)
            loss.backward()
            optimizer.step()
            losses_per_batch['train'] += loss.item()

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch, mask_batch in dataloaders['val']:
                x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)
                loss = model.neg_log_likehood(x_batch, y_batch, mask_batch)
                losses_per_batch['val'] += loss.item()

        for mode in ['train', 'val']:
            losses_per_batch[mode] /= len(dataloaders[mode].dataset)
            losses[mode].append(losses_per_batch[mode])

        if best_loss is None or best_loss > losses_per_batch['val']:
            best_loss = losses_per_batch['val']
            best_model_wts = model.state_dict()

        if verbose:
            print(
                'Epoch: {}'.format(epoch),
                'train_loss: {}'.format(losses_per_batch['train']),
                'val_loss: {}'.format(losses_per_batch['val']),
                sep=', '
            )

    model.load_state_dict(best_model_wts)
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
