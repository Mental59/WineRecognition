import re
import json
import numpy as np


data_info = json.load(open(r"G:\PythonProjects\WineRecognition2\data_info.json"))

NUMBER_KEYS = data_info['keys']['numerical']
WORD_KEYS = data_info['keys']['string']


def isnumber(word):
    try:
        float(word)
        return True
    except ValueError:
        return False


def getprob(freq_dict, key, word):
    try:
        return float(freq_dict[key].loc[freq_dict[key]['value'].str.lower() == word.lower()].iloc[0]['frequency'])
    except IndexError:
        return 0.0


def getprob_binary(freq_dict, key, word):
    return bool(len(freq_dict[key].loc[freq_dict[key]['value'].str.lower() == word.lower()]))


def calculate_probs(words: str, isnumber: bool, prob_func, freq_dict):
    probs = dict.fromkeys(NUMBER_KEYS + WORD_KEYS, 0.0)

    for key in NUMBER_KEYS if isnumber else WORD_KEYS:
        prob = prob_func(freq_dict, key, str(float(words))) if key == 'Add_BottleSize' else \
            prob_func(freq_dict, key, words)

        if re.match(r"[`']", words) and prob == 0.0:
            prob = prob_func(freq_dict, key, re.sub(r"[`']", "", words))

        probs[key] = prob

    return probs


def word2features(sent, i: int, freq_dict):
    word, label = sent[i]

    is_number = isnumber(word)

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'isNumber(word)': is_number
    }

    probs = calculate_probs(word, is_number, getprob_binary, freq_dict)
    for key in probs:
        features[key] = probs[key]

    if i > 0:
        word1, label1 = sent[i - 1]

        probs1 = calculate_probs(f'{word1} {word}', False, getprob_binary, freq_dict)
        for key in probs1:
            features[f'-1:BGram.{key}'] = probs1[key]

        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.label': label1,
        })

    if i < len(sent) - 1:
        word1, label1 = sent[i + 1]

        probs1 = calculate_probs(f'{word} {word1}', False, getprob_binary, freq_dict)
        for key in probs1:
            features[f'+1:BGram.{key}'] = probs1[key]

        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.label': label1,
        })

    return features


def sent2features(sent, freq_dict):
    return [word2features(sent, i, freq_dict) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent):
    return [token for token, label in sent]
