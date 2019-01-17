import os
import sys
import numpy as np

lib_path = os.path.abspath(os.path.join(__file__, './utils'))
sys.path.append(lib_path)

from utils.global_vars import character_to_index_mapping, NUM_CHARS


def index_to_one_hot(index):
    return np.eye(NUM_CHARS)[index]


def character_to_one_hot(chars):
    assert isinstance(chars, tuple) or isinstance(chars, list)
    assert len(chars) == 3
    if all(isinstance(element, tuple) for element in chars):
        ret_list = [np.concatenate([index_to_one_hot(character_to_index_mapping[char])
                          if char != " " else np.zeros((NUM_CHARS,)) for char in triplet]) for triplet in zip(*chars)]
        return np.stack(ret_list)
    return np.expand_dims(np.concatenate([index_to_one_hot(character_to_index_mapping[char])
                     if char != " " else np.zeros((NUM_CHARS,)) for char in chars]), 0)


if __name__ == '__main__':
    print(character_to_one_hot(('a', ',', ' ')))
    print(list(character_to_index_mapping.keys())[0])
