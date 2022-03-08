import re
from typing import Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset
from num2words import num2words
from features import features


class CustomDataset(Dataset):
    """
    CustomDataset
    returns sentence, tags, mask, custom_features if freq_dict was provided
    """

    num_regex = re.compile(r"^\d+(\.\d+)?$")
    fractional_number_regex = re.compile(r"^\d+/\d+$")

    def __init__(
            self,
            data,
            tag_to_ix,
            word_to_ix,
            freq_dict: Dict[str, pd.DataFrame] = None,
            case_sensitive=True,
            prepare_dataset=True,
            convert_nums2words=False
    ):
        super(CustomDataset, self).__init__()
        self.case_sensitive = case_sensitive
        self.convert_nums2words = convert_nums2words
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
        self.raw_data_cached = list(self.compute_raw_data(data))
        self.data = self.prepare_dataset() if prepare_dataset else None
        if self.freq_dict is not None:
            self.custom_features = self.compute_custom_features()

    def __getitem__(self, index):
        sentence, tags = self.data[index]
        mask = tags >= 0
        f = (self.custom_features[index] if self.custom_features is not None
             else torch.empty(0))
        return sentence, tags, mask, f

    def __len__(self):
        return len(self.data)

    def prepare_dataset(self):
        prepared_data = []
        max_len = self.compute_max_sentence_len(self.raw_data_cached)
        for sentence, tags in self.raw_data_cached:
            sentence = self.sentence_to_indices(sentence)
            tags = self.tags_to_indices(tags)

            sentence.extend([self.word_to_ix[self.pad]] * (max_len - len(sentence)))
            tags.extend([-1] * (max_len - len(tags)))

            prepared_data.append(
                (torch.LongTensor(sentence), torch.LongTensor(tags))
            )
        return prepared_data

    def sentence_to_indices(self, sentence):
        return [
            self.word_to_ix[word] if word in self.word_to_ix else self.word_to_ix[self.unk] for word in sentence
        ]

    def tags_to_indices(self, tags):
        return [self.tag_to_ix[tag] for tag in tags]

    @staticmethod
    def number2words(number) -> List[str]:
        return (
            num2words(number)
            .replace('-', ' ')
            .replace(',', '')
            .split()
        )

    def compute_raw_data(self, data):
        for sentence, tags in data:
            if not self.case_sensitive:
                sentence = [word.lower() for word in sentence]

            if self.convert_nums2words:
                new_sentence = []
                new_tags = []

                for index, (word, tag) in enumerate(zip(sentence, tags)):
                    if CustomDataset.num_regex.match(word):
                        words = CustomDataset.number2words(word)
                        new_sentence.extend(words)
                        new_tags.extend([tag] * len(words))
                    elif CustomDataset.fractional_number_regex.match(word):
                        first_price, second_price = word.split('/')
                        first_price_words = CustomDataset.number2words(first_price)
                        second_price_words = CustomDataset.number2words(second_price)

                        new_sentence.extend(first_price_words)
                        new_tags.extend([tag] * len(first_price_words))

                        new_sentence.append('/')
                        new_tags.append('Other')

                        new_sentence.extend(second_price_words)
                        new_tags.extend([tag] * len(second_price_words))
                    else:
                        new_sentence.append(word)
                        new_tags.append(tag)

                sentence = new_sentence
                tags = new_tags

            yield sentence, tags

    def raw_data(self):
        return self.raw_data_cached

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

    def compute_custom_features(self) -> List[torch.Tensor]:
        custom_features = []
        max_len = self.compute_max_sentence_len(self.raw_data_cached)
        for index, (sentence, tags) in enumerate(self.raw_data_cached):
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
