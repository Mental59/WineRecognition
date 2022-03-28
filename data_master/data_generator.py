import sys
import json
from typing import Dict
import re
import string
from random import choice
from tqdm import tqdm
import pandas as pd
from .complex_generator import *


class DataGenerator:
    """class for generating data structures"""

    regex = re.compile(
        r'(^[%s]*)|([%s]*$)' % (re.escape(string.punctuation), re.escape(string.punctuation))
    )

    @staticmethod
    def generate_freq_dict(*data: pd.DataFrame, keys: list, byword=False) -> Dict:

        def add_value(freq_dict, key, val, count, frequency):
            """
            Add value to frequency dictionary with given
            key frequency and value count
            """

            try:
                same = freq_dict[key].index(
                    next(
                        filter(lambda x: x['value'] == val, freq_dict[key])
                    )
                )
                freq_dict[key][same]['count'] += count
                freq_dict[key][same]['frequency'] += frequency

            except StopIteration:
                freq_dict[key].append(
                    dict(value=val, count=count, frequency=frequency)
                )

        all_values = {}
        for d in data:
            for key in filter(lambda key: key in d.columns, keys):
                if key not in all_values.keys():
                    all_values[key] = []
                all_values[key].extend(d[key].values.tolist())

        frequency_dictionary = {}

        for key in tqdm(all_values, desc='Generating frequency dictionary', file=sys.stdout):
            frequency_dictionary[key] = []

            for unique_value in set(all_values[key]):
                count = all_values[key].count(unique_value)
                frequency = round(count * 100 / len(all_values[key]), 4)

                unique_value_split = unique_value.split(', ') if ', ' in unique_value \
                    else unique_value.split('; ')

                for val in unique_value_split:
                    val = val.strip()

                    add_value(
                        freq_dict=frequency_dictionary,
                        key=key,
                        val=val,
                        count=count,
                        frequency=frequency
                    )

                    val_split = val.split()
                    if byword and len(val_split) > 1:
                        for word in val_split:
                            word = DataGenerator.regex.sub('', word)
                            add_value(
                                freq_dict=frequency_dictionary,
                                key=key,
                                val=word,
                                count=count,
                                frequency=frequency
                            )

            frequency_dictionary[key] = pd.DataFrame(frequency_dictionary[key])

        return frequency_dictionary

    @staticmethod
    def __process_entry(row, column, write_labels: bool, prepr_func=None) -> str:
        res = []
        for word in row.split():
            word_processed = False
            for symbol in word:
                if symbol in string.punctuation:
                    if symbol in ['&']:
                        res.append(f'{symbol} {column}\n' if write_labels else f'{symbol}\n')
                    else:
                        res.append(f'{symbol} Punctuation\n' if write_labels else f'{symbol}\n')
                else:
                    break
            word_removed_punctuations = DataGenerator.regex.sub('', word)

            if word_removed_punctuations:
                if prepr_func is not None:
                    word_removed_punctuations = prepr_func(word_removed_punctuations)
                res.append(
                    f'{word_removed_punctuations} {column}\n' if write_labels
                    else f'{word_removed_punctuations}\n'
                )
                word_processed = True

            if word_processed:
                for symbol in word[::-1]:
                    if symbol in string.punctuation:
                        res.append(f'{symbol} Punctuation\n' if write_labels else f'{symbol}\n')
                    else:
                        break
        return ''.join(res)

    @staticmethod
    def generate_data_text(data: pd.DataFrame, keys: list, write_labels=True) -> str:
        r = re.compile(
            r'(^[%s]*)|([%s]*$)' % (';,', ';,')
        )
        res = []
        for _, row in tqdm(data.iterrows()):
            for column in filter(lambda key: key in data.columns, keys):
                if pd.isna(row[column]):
                    continue
                row_column = r.sub('', str(row[column]))
                if row_column:
                    res.append(DataGenerator.__process_entry(row_column, column, write_labels))
            res.append('\n')
        return ''.join(res)

    @staticmethod
    def generate_data_text_complex(data: pd.DataFrame, complex_generator: ComplexGeneratorMain,
                                   write_labels=True) -> str:
        """
        :param data: pandas DataFrame
        :param complex_generator: iterable object which returns columns of provided data
        :param write_labels: write labels or not
        :return: string
        """

        r = re.compile(
            r'(^[%s]*)|([%s]*$)' % (';,', ';,')
        )
        res = []
        for _, row in tqdm(data.iterrows()):
            for column in complex_generator:
                if pd.isna(row[column]):
                    continue
                row_column = r.sub(
                    '',
                    str(row[column])
                )
                if row_column:
                    res.append(DataGenerator.__process_entry(row_column, column, write_labels))
            res.append('\n')
        return ''.join(res)

    @staticmethod
    def generate_data_text_menu(data: pd.DataFrame, complex_generator: ComplexGeneratorMenu,
                                write_labels=True) -> str:
        """
        :param data: pandas DataFrame
        :param complex_generator: iterable object which returns columns of provided data
        :param write_labels: write labels or not
        :return: string
        """

        r = re.compile(
            r'(^[%s]*)|([%s]*$)' % (';,', ';,')
        )
        res = []
        for _, row in tqdm(data.iterrows()):
            for column, prepr_func, values in complex_generator:
                if values is not None:
                    res.append(f'{choice(values)} {column}\n')
                else:
                    if pd.isna(row[column]):
                        continue
                    row_column = r.sub(
                        '',
                        str(row[column])
                    )
                    if row_column:
                        res.append(DataGenerator.__process_entry(row_column, column, write_labels, prepr_func))
            res.append('\n')
        return ''.join(res)

    @staticmethod
    def generate_sents(lines: list):
        sents = []
        example = []
        for line in lines:
            if not line:
                if example:
                    sents.append(example)
                    example = []
                continue

            example.append(tuple(line.split()[:2]))

        return sents

    @staticmethod
    def generate_sents2(lines: list):
        sents = []
        sentences, tags = [], []
        for line in lines:
            if not line:
                if sentences and tags:
                    sents.append((sentences, tags))
                    sentences, tags = [], []
                continue

            sentence, *tag = line.split()
            tag = ' '.join(tag)
            sentences.append(sentence)
            tags.append(tag)

        return sents

    @staticmethod
    def generate_probability_table(marginals, sents) -> pd.DataFrame:
        """Generates probability table from marginals for each word"""
        df = pd.DataFrame(marginals).applymap(
            lambda d: sorted(d.items(), key=lambda x: x[1], reverse=True) if d is not None else None
        ).applymap(
            lambda l: [(x[0], round(x[1], 4)) for x in l] if l is not None else None
        )
        for (i, df_row), (x_row, _) in zip(df.iterrows(), sents):
            for j, word in enumerate(x_row):
                df.iloc[i, j] = json.dumps({word: df.iloc[i, j]})
        return df
