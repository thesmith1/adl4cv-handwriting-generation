import numpy as np
from global_vars import character_to_index_mapping, NUM_CHAR


def character_to_index(char):
    ret = []
    if isinstance(char, tuple):
        for el in char:
            ret.append(character_to_index_mapping[el])
    else:
        ret.append(character_to_index_mapping[char])
    return ret


def index_to_one_hot(idx):
    ret = np.zeros((NUM_CHAR, len(idx)))
    for i, el in enumerate(idx):
        ret[el, i] = 1
    return ret


def character_to_one_hot(char):
    return index_to_one_hot(character_to_index(char))


if __name__ == '__main__':
    print(character_to_one_hot(('a', ',', ' ')))
