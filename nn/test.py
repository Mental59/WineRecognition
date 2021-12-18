from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train, CustomDataset, plot_losses
from model import BiLSTM_CRF
from data_master import DataLoader as data_loader


def main():
    device = 'cuda'
    tag_to_ix = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    word_to_ix = {
        'red': 0,
        'word2': 1,
        'word3': 2,
        'word4': 3,
        'word5': 4,
        'word6': 5,
        'word7': 6,
        'PAD': 7,
        'UNK': 8
    }
    vocab_size = len(word_to_ix)
    embedding_dim = 8
    hidden_dim = 6
    num_epochs = 5
    freq_dict = data_loader.load_frequency_dictionary(r'G:\PythonProjects\WineRecognition2\data\dictionaries\Dict-byword_Halliday_Winesearcher_Wine_AU-only_completed_rows')

    training_data = [
        ('word1 word3 word5'.split(), 'A C E'.split()),
        ('word7 word2 word7 word1'.split(), 'A C D E'.split()),
        ('word2 word3 word1 word7 word6'.split(), 'A B E D C'.split()),
        ('word2 word7 word6'.split(), 'B A E'.split())
    ]
    val_data = [
        ('word3 word5'.split(), 'A E'.split()),
        ('word7 word2 word5 word3 word1'.split(), 'A B C D E'.split()),
        ('word2 word3 word1 word3 word4'.split(), 'A B E D C'.split()),
        ('word1 word5 word2'.split(), 'B C E'.split())
    ]

    train_dataset = CustomDataset(training_data, tag_to_ix, word_to_ix, freq_dict=freq_dict)
    val_dataset = CustomDataset(val_data, tag_to_ix, word_to_ix, freq_dict=freq_dict)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False),
        'val': DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=False)
    }

    model = BiLSTM_CRF(vocab_size, len(tag_to_ix), embedding_dim, hidden_dim, custom_features_size=43).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    model, losses = train(
        model,
        optimizer,
        dataloaders,
        device,
        num_epochs,
        '',
        tqdm=tqdm
    )
    plot_losses(losses)


if __name__ == '__main__':
    main()
