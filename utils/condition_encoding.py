import numpy as np
from global_vars import character_to_index_mapping, NUM_CHARS


def character_to_index(char):
    ret = []
    if isinstance(char, tuple):
        for el in char:
            ret.append(character_to_index_mapping[el])
    else:
        ret.append(character_to_index_mapping[char])
    return ret


def index_to_one_hot(idx):
    ret = np.zeros((len(idx), NUM_CHARS))
    for i, el in enumerate(idx):
        ret[i, el] = 1
    return ret


def character_to_one_hot(char):
    return index_to_one_hot(character_to_index(char))


if __name__ == '__main__':
    print(character_to_one_hot(('a', ',', ' ')))
    print(list(character_to_index_mapping.keys())[0])
