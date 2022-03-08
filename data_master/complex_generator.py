import random
from typing import Dict, Tuple, Callable, List

__all__ = [
    'ComplexGeneratorBase',
    'ComplexGeneratorMenu',
    'ComplexGeneratorMain',
    'CfgValue'
]


class CfgValue:
    def __init__(self, column: str, prob: float = 1.0, prepr_func: Callable = None, values: List[str] = None):
        self.column = column
        self.prob = prob
        self.prepr_func = prepr_func
        self.values = values


class ComplexGeneratorBase:
    def __iter__(self):
        raise NotImplementedError


class ComplexGeneratorMenu(ComplexGeneratorBase):
    def __init__(self, cfg: List[CfgValue]):
        self.cfg = cfg

    def __iter__(self):
        for cfg_value in self.cfg:
            if cfg_value.prob > random.random():
                yield cfg_value.column, cfg_value.prepr_func, cfg_value.values


class ComplexGeneratorMain(ComplexGeneratorBase):
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
