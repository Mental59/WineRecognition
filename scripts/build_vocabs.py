import os
import sys
import json
import re
from tqdm import tqdm
import string


current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.split(current_dir)[0]
if upper_dir not in sys.path:
    sys.path.insert(0, upper_dir)


from data_master import DataLoader


dictionary_paths_vocab_names = [
    (
        'G:/PythonProjects/WineRecognition2/data/dictionaries/Dict-byword_Halliday_Wine_AU-only_completed_rows',
        'Words_Halliday_Wine_AU'
    ),
    (
        'G:/PythonProjects/WineRecognition2/data/dictionaries/Dict-byword_Halliday_WineSearcher_Wine_AU-only_completed_rows',
        'Words_Halliday_WineSearcher_Wine_AU'
    ),
    (
        r'G:\PythonProjects\WineRecognition2\data\dictionaries\Dict-byword_Halliday_Winesearcher_Bruxelles',
        'Words_Halliday_WineSearcher_Bruxelles'
    )
]

regex = re.compile(r'(^[%s]*)|([%s]*$)' % (re.escape(string.punctuation), re.escape(string.punctuation)))

num_words = [
    'and', 'eight', 'eighteen', 'eighty', 'eleven',
    'fifteen', 'fifty', 'five', 'forty', 'four', 'fourteen', 'hundred',
    'nine', 'nineteen', 'ninety', 'one', 'seven', 'seventeen',
    'seventy', 'six', 'sixteen', 'sixty', 'ten', 'thirteen',
    'thirty', 'thousand', 'thousand,', 'three', 'twelve',
    'twenty', 'two', 'zero', 'million'
]

for dictionary_path, vocab_name in dictionary_paths_vocab_names:

    DICT = DataLoader.load_frequency_dictionary(dictionary_path)

    vocab = {}

    i = 0
    for key in tqdm(DICT):
        for word in filter(lambda x: len(x.split()) == 1, DICT[key].value.values):
            word = regex.sub('', word)
            if word not in vocab:
                vocab[word] = i
                i += 1

    for p in string.punctuation:
        if p not in vocab:
            vocab[p] = i
            i += 1

    for w in ['PAD', 'UNK']:
        vocab[w] = i
        i += 1

    with open(os.path.join('G:/PythonProjects/WineRecognition2/data/vocabs/', f'{vocab_name}.json'), 'w', encoding='utf-8') as file:
        json.dump(vocab, file)

    for w in num_words:
        if w not in vocab:
            vocab[w] = i
            i += 1

    with open(os.path.join('G:/PythonProjects/WineRecognition2/data/vocabs/', f'{vocab_name}_WORD_NUMS.json'), 'w', encoding='utf-8') as file:
        json.dump(vocab, file)

    print('Vocab length:', len(vocab))
