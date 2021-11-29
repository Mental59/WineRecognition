import os
import numpy as np

DATA_PATH = r'G:\PythonProjects\WineRecognition2\data\menus\Wines.txt'
OUTPUT_PATH = r'G:\PythonProjects\WineRecognition2\data\text'
FILENAME = 'menu_txt.txt'


def add_white_spaces_between_punctuation(s: str):
    chars = list(s)
    indices = np.array([i for i, c in enumerate(chars) if c in ',;'])
    for i in indices:
        left = next(iter(chars[i-1:i]), '')
        right = next(iter(chars[i + 1:i + 2]), '')
        if left != ' ':
            chars.insert(i, ' ')
            indices += 1
            i += 1
        if right != ' ':
            chars.insert(i + 1, ' ')
            indices += 1

    return ''.join(chars)


def main():
    tag = 'UNKNOWN'
    sents = []
    for line in open(DATA_PATH, encoding='utf-8'):
        line = line.strip()
        if line.startswith('http') or not line:
            continue
        line = add_white_spaces_between_punctuation(line)
        sent = '\n'.join(' '.join([word, tag]) for word in line.split()) + '\n'
        sents.append(sent)

    with open(os.path.join(OUTPUT_PATH, FILENAME), 'w', encoding='utf-8') as file:
        file.write('\n'.join(sents))


if __name__ == '__main__':
    main()
