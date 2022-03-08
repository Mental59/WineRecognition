from collections import Counter
import numpy as np

__all__ = [
    'count_unk_foreach_tag',
    'compute_model_confidence'
]


def count_unk_foreach_tag(X_test, y_true, classes, unk_index):
    unk_counter = Counter()
    word_counter = Counter()

    for sentence, tags in zip(X_test, y_true):
        for word_ix, tag in zip(sentence, tags):
            word_counter[tag] += 1
            if word_ix == unk_index:
                unk_counter[tag] += 1

    res = dict(unk_counter)
    for tag in set(classes + list(res.keys())):
        if tag not in res:
            res[tag] = 0
        else:
            res[tag] /= word_counter[tag]

    return res


def compute_model_confidence(marginals) -> list:
    return [np.mean([max(d.values()) for d in marginal]) for marginal in marginals]
