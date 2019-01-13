# crop utility
rectangle_shape = (420, 140)  # first param. must be tuned to match the height of two lines in a college-ruled notebook

# pre-processing
SUP_REMOVE_WIDTH = 25
INF_REMOVE_WIDTH = 105

# training
save_every = 100
add_character_every = 100
produce_every = 100
num_characters_to_generate = 5
random_characters_to_generate = 'ABCDE'
D_loss_threshold = 0.3
G_loss_threshold = 0.3

# network
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
NOISE_LENGTH = 100
character_to_index_mapping = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
    'L': 11,
    'M': 12,
    'N': 13,
    'O': 14,
    'P': 15,
    'Q': 16,
    'R': 17,
    'S': 18,
    'T': 19,
    'U': 20,
    'V': 21,
    'W': 22,
    'X': 23,
    'Y': 24,
    'Z': 25,
    'a': 26,
    'b': 27,
    'c': 28,
    'd': 29,
    'e': 30,
    'f': 31,
    'g': 32,
    'h': 33,
    'i': 34,
    'j': 35,
    'k': 36,
    'l': 37,
    'm': 38,
    'n': 39,
    'o': 40,
    'p': 41,
    'q': 42,
    'r': 43,
    's': 44,
    't': 45,
    'u': 46,
    'v': 47,
    'w': 48,
    'x': 49,
    'y': 50,
    'z': 51,
    '0': 52,
    '1': 53,
    '2': 54,
    '3': 55,
    '4': 56,
    '5': 57,
    '6': 58,
    '7': 59,
    '8': 60,
    '9': 61,
    '.': 62,
    ',': 63,
    '?': 64,
    '!': 65,
    ';': 66,
    ':': 67,
    '-': 68,
    '\'': 69
}

NUM_CHARS = len(character_to_index_mapping)
