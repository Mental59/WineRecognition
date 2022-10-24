import os
import json
from glob import glob
from tqdm import tqdm
import re
from nltk.tokenize import word_tokenize

LABELS_PATH = 'data/results/labels'
MENUS_PATH = 'data/results/menus'


def load_annotation(path):
    with open(path, encoding='utf-8') as file:
        return json.load(file)


def filter_cond(word):
    return not re.search(r"wine-searcher|\".*\"|'.*'|`.*`", word)


def split_number(word: str):
    match = re.search(r"\d+", word)
    if match is None:
        return [word]
    number = match.group()
    return re.sub(number, number + ' ', word).strip().split()


def get_words_from_annotation(annotation):
    words = []
    for word in [word for word in word_tokenize(annotation[0]) if filter_cond(word)]:
        words.extend(split_number(word))
    return words


def save_dataset(words, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        file.writelines([(f'{word} UNKNOWN\n' if word else '\n') for word in words])


def main():
    labels_annotation_words = []
    menus_annotation_words = []

    for path in tqdm(glob(os.path.join(LABELS_PATH, '*.json'))):
        annotation = load_annotation(path)
        words = get_words_from_annotation(annotation)
        labels_annotation_words.extend(words + [''])
    labels_annotation_words.pop()

    for path in tqdm(glob(os.path.join(MENUS_PATH, '*.json'))):
        annotation = load_annotation(path)
        words = get_words_from_annotation(annotation)
        menus_annotation_words.extend(words + [''])
    menus_annotation_words.pop()

    save_dataset(labels_annotation_words, 'data/results/wine_labels.txt')
    save_dataset(menus_annotation_words, 'data/results/wine_menus.txt')


if __name__ == '__main__':
    main()
