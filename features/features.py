import re
from typing import Callable


def isnumber(word: str):
    try:
        float(word)
        return True
    except ValueError:
        return False


def getprob(freq_dict, key: str, word: str) -> float:
    try:
        return float(freq_dict[key].loc[word.lower(), 'frequency'])
    except (IndexError, KeyError):
        return 0.0


def getprob_binary(freq_dict, key, word) -> bool:
    try:
        return not freq_dict[key].loc[word.lower()].empty
    except KeyError:
        return False


def calculate_probs(word: str, prob_func: Callable, freq_dict):
    probs = dict.fromkeys(freq_dict.keys(), 0.0)

    for key in freq_dict:
        prob = prob_func(freq_dict, key, word)

        if re.match(r"[`']", word) and not prob:
            prob = prob_func(freq_dict, key, re.sub(r"[`']", "", word))

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

    probs = calculate_probs(word, getprob_binary, freq_dict)
    for key in probs:
        features[key] = probs[key]

    if i > 0:
        word1, label1 = sent[i - 1]

        probs1 = calculate_probs(f'{word1} {word}', getprob_binary, freq_dict)
        for key in probs1:
            features[f'-1:BGram.{key}'] = probs1[key]

        features.update({
            '-1:word.lower()': word1.lower()
        })

    if i < len(sent) - 1:
        word1, label1 = sent[i + 1]

        probs1 = calculate_probs(f'{word} {word1}', getprob_binary, freq_dict)
        for key in probs1:
            features[f'+1:BGram.{key}'] = probs1[key]

        features.update({
            '+1:word.lower()': word1.lower()
        })

    return features


def sent2features(sent, freq_dict):
    return [word2features(sent, i, freq_dict) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent):
    return [token for token, label in sent]


def sent2features_generator(sent, freq_dict):
    return (word2features(sent, i, freq_dict) for i in range(len(sent)))


def sent2labels_generator(sent):
    return (label for token, label in sent)


def sent2tokens_generator(sent):
    return (token for token, label in sent)


if __name__ == '__main__':
    from data_master import DataLoader
    import time
    freq_dict = DataLoader.load_frequency_dictionary(
        r'G:\PythonProjects\WineRecognition2\data\dictionaries\Dict-byword_Halliday_Winesearcher_Wine_AU-only_completed_rows',
        to_lowercase=True
    )

    elapsed = time.time()
    print(getprob_binary(freq_dict, 'Add_Brand', 'the'))
    elapsed = time.time() - elapsed
    print(elapsed)
