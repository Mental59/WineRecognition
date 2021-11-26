import os
import sys
from typing import Dict, Tuple
from random import choice
from tqdm import tqdm
import re
import pandas as pd
import random
import string
import matplotlib.pyplot as plt
from matplotlib import style


class ComplexGenerator:
    def __iter__(self):
        raise NotImplementedError


class ComplexGeneratorMenu(ComplexGenerator):
    def __init__(self, cfg):
        self.cfg = cfg

    def __iter__(self):
        for column_name, prob, prepr_func, values in self.cfg:
            if prob > random.random():
                yield column_name, prepr_func, values


class ComplexGeneratorMain(ComplexGenerator):
    """class for generating columns with given cfg, which contains probabilities"""

    def __init__(self, cfg: Dict[str, Tuple[str, float]]):
        self.cfg = cfg
        self.packs = [
            [cfg['producer'], cfg['brand']],
            [cfg['keywordTrue'], cfg['keywordFalse'], cfg['grape'], cfg['region']],
            [cfg['type'], cfg['color'], cfg['sweetness'], cfg['closure'], cfg['bottleSize']],
            [cfg['certificate'], cfg['price']],
        ]

    def __generate_keys(self):
        keys = []

        for i in 1, 2:
            random.shuffle(self.packs[i])

        for pack in self.packs:
            keys.extend(pack)

        if random.random() < 0.5:
            keys.insert(-1, self.cfg['vintage'])
        else:
            keys.insert(0, self.cfg['vintage'])

        return keys

    def __iter__(self):
        for col, chance in self.__generate_keys():
            if chance > random.random():
                yield col


class DataLoader:
    """class for loading data with needed format"""

    @staticmethod
    def load_frequency_dictionary(path: str) -> Dict:
        freq_dict = {}

        for filename in os.listdir(path):
            key = os.path.splitext(filename)[0]
            freq_dict[key] = pd.read_csv(f'{os.path.join(path, filename)}', dtype=object)
            freq_dict[key].fillna('', inplace=True)

        return freq_dict

    @staticmethod
    def load_csv_data(path: str, fillna=True) -> pd.DataFrame:
        data = pd.read_csv(path, dtype=object)

        if fillna:
            data.fillna('', inplace=True)

        return data

    @staticmethod
    def load_excel_data(path: str, fillna=True) -> pd.DataFrame:
        data = pd.read_excel(path)

        if fillna:
            data.fillna('', inplace=True)

        return data

    @staticmethod
    def preprocess(
            data: pd.DataFrame,
            fill_bottle_size: float = None,
            only_completed_rows=False,
            drop_not_add_columns=False
    ) -> pd.DataFrame:

        new_data = data.copy()

        if fill_bottle_size:
            new_data.loc[new_data['Add_BottleSize'] == '', 'Add_BottleSize'] = fill_bottle_size

        if only_completed_rows:
            new_data = new_data.loc[new_data['IsCompleted'] == True]

        if drop_not_add_columns:
            new_data.drop(
                columns=[col for col in new_data.columns if not col.startswith('Add')],
                inplace=True
            )

        new_data = new_data.applymap(
            lambda x: ''.join(
                char for char in str(x).replace('} {', '; ') if char not in '{}[]'
            )
        )

        vintage = 'Add_Vintage'
        if vintage in new_data.columns:
            new_data.loc[(new_data[vintage] == 'Non Vintage') | (new_data[vintage] == 'Nv'), vintage] = ''

        return new_data


class DataGenerator:
    """class for generating data structures"""

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
                            add_value(
                                freq_dict=frequency_dictionary,
                                key=key,
                                val=word,
                                count=count,
                                frequency=frequency
                            )

            frequency_dictionary[key] = pd.DataFrame(frequency_dictionary[key])

        return frequency_dictionary

    regex = re.compile(
        r'(^[%s]*)|([%s]*$)' % (re.escape(string.punctuation), re.escape(string.punctuation))
    )

    @staticmethod
    def __process_entry(row, column, write_labels: bool, prepr_func=None) -> str:
        res = []
        for word in row.split():
            word_processed = False
            for symbol in word:
                if symbol in string.punctuation:
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
                sents.append(example)
                example = []
                continue

            example.append(tuple(line.split()[:2]))

        return sents[:-1]

    @staticmethod
    def generate_sents2(lines: list):
        sents = []
        sentences, tags = [], []
        for line in lines:
            if not line:
                sents.append((sentences, tags))
                sentences, tags = [], []
                continue

            sentence, *tag = line.split()
            tag = ' '.join(tag)
            sentences.append(sentence)
            tags.append(tag)

        return sents[:-1]


class DataSaver:
    """class for saving generated data structures"""

    @staticmethod
    def save_frequency_dictionary(frequency_dictionary: Dict, excel_path, csv_folder):
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            for key in frequency_dictionary:
                frequency_dictionary[key].sort_values(by='count', inplace=True, ascending=False)
                frequency_dictionary[key].to_excel(writer, sheet_name=key)
                frequency_dictionary[key].to_csv(os.path.join(csv_folder, f'{key}.csv'), index=False)


class DataAnalyzer:
    """Class for analyzing results"""

    @staticmethod
    def analyze(test_eval, keys, table_save_path, diagram_save_path):
        """
        Creates colored table and save it on path "table_save_path" (.xlsx)
        Creates bar chart and save it on path "diagram_save_path" (.png)
        """

        predicted = []
        actual = []

        for eval in test_eval:

            predicted_example = dict.fromkeys(keys, '')
            actual_example = dict.fromkeys(keys, '')

            for word, true_tag, pred_tag in eval:
                if true_tag in actual_example.keys():
                    actual_example[true_tag] += f'{word} '

                if pred_tag in predicted_example.keys():
                    predicted_example[pred_tag] += f'{word} '

            actual.append({key: value.rstrip() for key, value in actual_example.items()})
            predicted.append({key: value.rstrip() for key, value in predicted_example.items()})

        df_actual = pd.DataFrame({key: [wine.get(key) for wine in actual] for key in keys})
        df_predicted = pd.DataFrame({key: [wine.get(key) for wine in predicted] for key in keys})

        matches = dict.fromkeys(keys + ['All'], 0)  # совпадения
        matched_indices = []
        false_negative = dict.fromkeys(keys, 0)  # ложноотрицательные ошибки
        false_negative_indices = []
        false_positive = dict.fromkeys(keys, 0)  # ложноположительные ошибки
        false_positive_indices = []

        for index, row in df_predicted.iterrows():

            flag_all = True

            for column in keys:

                if row[column] == df_actual.iloc[index][column]:

                    matches[column] += 1

                    matched_indices.append((index, column))
                else:

                    flag_all = False

                    if df_actual.iloc[index][column]:

                        false_negative[column] += 1

                        false_negative_indices.append((index, column))
                    else:
                        false_positive[column] += 1

                        false_positive_indices.append((index, column))

            if flag_all:
                matches['All'] += 1

        for key in matches:

            if key == 'All': continue

            false_positive[key] += matches[key]

            false_negative[key] += false_positive[key]

        style.use('seaborn-darkgrid')

        fig = plt.figure(figsize=(18, 8))

        index = list(range(1, len(matches) + 1))

        index2 = list(range(1, len(false_positive) + 1))

        analyze_res = dict(sorted(matches.items(), key=lambda x: x[1], reverse=True))

        false_positive = {key: false_positive[key] for key in analyze_res if key != 'All'}

        false_negative = {key: false_negative[key] for key in analyze_res if key != 'All'}

        plt.barh(index2, false_negative.values(), 0.6, label='Кол-во ложноотрицательных ошибок', color='tab:blue')
        plt.barh(index2, false_positive.values(), 0.6, label='Кол-во ложноположительных ошибок', color='tab:orange')
        plt.barh(index, analyze_res.values(), 0.6, label='Кол-во совпадений', color='tab:green')

        plt.title('Результаты')

        plt.yticks(index, analyze_res.keys())

        fig.legend(loc='upper right')

        for index, (key, value) in enumerate(analyze_res.items()):

            if value >= 100: plt.text(0, index + .8, str(value))

            if key != 'All':

                false_negative[key] -= false_positive[key]

                false_positive[key] -= value

                if false_positive[key] >= 100:
                    plt.text(value, index + .8, str(false_positive[key]))

                if false_negative[key] >= 100:
                    plt.text(value + false_positive[key], index + .8, str(false_negative[key]))

        def set_colors(data):
            attr = 'background-color: {};border-width: thin'

            res = data.copy()

            for index, column in matched_indices:
                res.iloc[index][column] = attr.format('green')

            for index, column in false_positive_indices:
                res.iloc[index][column] = attr.format('orange')

            for index, column in false_negative_indices:
                res.iloc[index][column] = attr.format('blue')

            return res

        colored_predicted = df_predicted.style.apply(set_colors, axis=None)

        with pd.ExcelWriter(table_save_path, engine='xlsxwriter') as writer:
            colored_predicted.to_excel(writer, sheet_name='predicted')
            df_actual.to_excel(writer, sheet_name='actual')

        plt.savefig(diagram_save_path)
