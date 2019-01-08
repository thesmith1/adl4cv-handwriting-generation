import numpy as np
from global_vars import character_to_index_mapping, NUM_CHARS


def index_to_one_hot(index):
    return np.eye(NUM_CHARS)[index]


def character_to_one_hot(chars):
    if not isinstance(chars, tuple) and not isinstance(chars, list):
        chars = (chars,)
    return np.stack([index_to_one_hot(character_to_index_mapping[char])
                     if char != " " else np.zeros((NUM_CHARS,)) for char in chars])


if __name__ == '__main__':
    print(character_to_one_hot(('a', ',', ' ')))
    print(list(character_to_index_mapping.keys())[0])
