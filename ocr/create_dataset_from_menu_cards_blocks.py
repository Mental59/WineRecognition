import os
import json
from glob import glob

DATA_PATH = r'G:\PythonProjects\WineRecognition2\ocr\data\results2\markup'
OUTPUT_PATH = './data/results2/wine_menus_blocks2.txt'


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_text_pages_from_file(file_path):
    with open(file_path, encoding='utf-8') as file:
        data = json.load(file)
        return data['fullTextAnnotation']['pages']


def symbols_to_word_string(symbols):
    word = [str(symbol['text']) for symbol in symbols]
    return ''.join(word)


def paragraph_to_words_string(paragraph):
    words = [symbols_to_word_string(word['symbols']) for word in paragraph['words']]
    return words


def create_text_block(block):
    text_blocks = [paragraph_to_words_string(paragraph) for paragraph in block['paragraphs']]
    return text_blocks


def create_text_page(page):
    text_page = [create_text_block(block) for block in page['blocks']]
    return text_page


def create_dataset(pages):
    dataset = [create_text_page(page) for page in pages]
    return dataset


def create_final_dataset(datasets):
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as file:
        for dataset in datasets:
            for block in dataset:
                for paragraph in block:
                    file.writelines([f'{word} UNKNOWN\n' for word in paragraph] + ['\n'])
                if len(block) > 1:
                    file.writelines([f'{word} UNKNOWN\n' for word in flatten(block)] + ['\n'])


def main():
    file_paths = glob(os.path.join(DATA_PATH, '*.json'))
    datasets = [create_dataset(get_text_pages_from_file(file_path))[0] for file_path in file_paths]
    create_final_dataset(datasets)


if __name__ == '__main__':
    main()
