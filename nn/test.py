from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train, CustomDataset, plot_losses
from model import BiLSTM_CRF


def main():
    device = 'cuda'
    tag_to_ix = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    word_to_ix = {
        'word1': 0,
        'word2': 1,
        'word3': 2,
        'word4': 3,
        'word5': 4,
        'word6': 5,
        'word7': 6,
        'PAD': 7
    }
    vocab_size = len(word_to_ix)
    embedding_dim = 8
    hidden_dim = 6
    num_epochs = 5

    training_data = [
        ('word1 word2 word3 word4 word5'.split(), 'A A B C E'.split()),
        ('word7 word2 word7 word3 word1'.split(), 'A A C D E'.split()),
        ('word2 word3 word1 word7 word6'.split(), 'A B E D C'.split()),
        ('word2 word5 word2 word7 word6'.split(), 'B B A A E'.split())
    ]
    val_data = [
        ('word3 word1 word3 word4 word5'.split(), 'A B D C E'.split()),
        ('word7 word2 word5 word3 word1'.split(), 'A B C D E'.split()),
        ('word2 word3 word1 word3 word4'.split(), 'A B E D C'.split()),
        ('word1 word5 word2 word3 word6'.split(), 'B C C A E'.split())
    ]

    train_dataset = CustomDataset(training_data, tag_to_ix, word_to_ix)
    val_dataset = CustomDataset(val_data, tag_to_ix, word_to_ix)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True, num_workers=1),
        'val': DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=True, num_workers=1)
    }

    model = BiLSTM_CRF(vocab_size, len(tag_to_ix), embedding_dim, hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    model, losses = train(model, optimizer, dataloaders, device, num_epochs, tqdm)
    plot_losses(losses)


if __name__ == '__main__':
    main()
