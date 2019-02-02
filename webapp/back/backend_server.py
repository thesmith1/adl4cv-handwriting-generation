import argparse
from os import listdir
from os.path import join, isfile
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import skimage
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from matplotlib.pyplot import imshow, show

from utils.image_utils import generate_optimized_from_string, CONTRAST_INCREASE
from utils.global_vars import character_to_index_mapping
from stitching import stitch

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

default_model_name = 'G_2019-01-22 22_33_20.613720_3600'
models_path = '../../data/models/'
assets_path = 'assets/'
encoding = 'utf-8'
display_width = 800  # pixels
characters_set = set(character_to_index_mapping.keys())
characters_set.add(' ')

state = {
    'words': [],
    'current_line': None,
    'text': ''
}

'''
Insertion requests:
- frontend sends request only after a timeout (~ms), such that a single request includes multiple characters
    Insertion distinguishes between two cases:
    - insert at the end (easy)
    - insert in the middle (tricky)
    The response of a non completely filled line of text is padded with black
    At a new request, the backend must generate again the last character, now with the correct conditioning
    If the user enters 'Enter', the line gets padded with black and a new line starts
    The backend must clean the input text of the characters that cannot be generated

Backend's local state:
- list of images of words (including prev and next spaces)
- image of current line (without black padding)
- string of transcription

Every request must contain the index of the character that is requested (the first)
'''


def ndarray_to_base64(arr: np.ndarray):
    with BytesIO() as output_bytes:
        arr = np.clip(((arr - np.min(arr)) / np.max(arr)) * 2 - 1, -1.0, 1.0)
        image = Image.fromarray(skimage.img_as_ubyte(arr))
        image.save(output_bytes, 'JPEG')
        bytes_data = output_bytes.getvalue()
    return str(base64.b64encode(bytes_data), encoding)


def black_pad(line: np.ndarray):
    pad_amount = display_width - line.shape[1]
    if pad_amount < 0:
        raise ValueError('Line too long')
    pad_arr = np.zeros((line.shape[0], pad_amount), dtype=np.float64)
    return np.concatenate([line, pad_arr], axis=1)


def flush_to_current_line(new_portion, style, index):
    if new_portion is not None:
        append_to_current_line(new_portion, style, index)
    completed_line = state['current_line']
    state['current_line'] = None
    return completed_line, index


def append_to_current_line(new_portion, style, index):
    if state['current_line'] is not None:
        last_char = state['text'][index - 1]
        if len(state['text']) > index + 1:
            first_new_char = state['text'][index + 1]
        else:
            first_new_char = ' '
        space_image = generate_optimized_from_string(g, last_char + ' ' + first_new_char, style,
                                                     CONTRAST_INCREASE, device=dev)[0]
        state['current_line'], _, _ = stitch(state['current_line'], space_image)
        state['current_line'], _, _ = stitch(state['current_line'], new_portion)
    else:
        state['current_line'] = new_portion


def get_current_line_width():
    if state['current_line'] is None:
        return 0
    else:
        return state['current_line'].shape[1]


@app.route('/')
def index():
    return 'Index Page!'


@app.route('/insert', methods=['POST'], strict_slashes=False)
@cross_origin()
def insert():
    """
    parameters: text, index, style
    """
    params = request.json
    ret = {}
    input_text = ''
    is_new_line = False
    completed_line = None
    # Clean text from unwanted characters
    for char in params['text']:
        if char in characters_set:
            input_text = input_text + char
    index = params['index']
    if index == len(state['text']):  # Insert at the end
        state['text'] = input_text
        if state['current_line'] is None:  # New line
            input_text = ' ' + input_text
        input_text = input_text + ' '  # Add a final space
        new_words = [' ' + word + ' ' for word in input_text[index:].split()]  # Isolate new words
        new_words_images = []
        for word in new_words:
            word_characters = generate_optimized_from_string(g, word, params['style'], CONTRAST_INCREASE, device=dev)
            word_image = word_characters[0]
            for i in range(len(word_characters) - 1):
                word_image, _, _ = stitch(word_image, word_characters[i + 1])  # Stitch characters of a word
            new_words_images.append(word_image)
        state['words'].extend(new_words_images)  # Update the state with the images of the new words
        new_portion = new_words_images[0]
        if get_current_line_width() + new_portion.shape[1] > display_width:
            is_new_line = True
            completed_line, index = flush_to_current_line(None, params['style'], index)
        for i in range(len(new_words_images) - 1):
            if get_current_line_width() + new_portion.shape[1] + new_words_images[i + 1].shape[1] > display_width:
                is_new_line = True
                completed_line, index = flush_to_current_line(new_portion, params['style'], index)
                new_portion = new_words_images[i + 1]
            else:
                prev_word = new_words[i]
                next_word = new_words[i + 1]
                prev_letter = prev_word[-2]
                next_letter = next_word[1]
                space_image = generate_optimized_from_string(g, prev_word[-2] + ' ' + next_word[1], params['style'],
                                                             CONTRAST_INCREASE, device=dev)[0]
                # imshow(space_image, cmap='Greys_r')
                # show()
                new_portion, _, _ = stitch(new_portion, space_image)
                new_portion, _, _ = stitch(new_portion, new_words_images[i + 1])
        append_to_current_line(new_portion, params['style'], index)
    else:
        # Insert in the middle
        pass
    # Build response
    ret['is_new_line'] = False
    if is_new_line:
        ret['is_new_line'] = True
        ret['completed_line'] = completed_line
        ret['completed_line'] = black_pad(ret['completed_line'])
        ret['completed_line'] = ndarray_to_base64(ret['completed_line'])
    ret['current_line'] = state['current_line']
    ret['current_line'] = black_pad(ret['current_line'])
    ret['current_line'] = ndarray_to_base64(ret['current_line'])
    response = jsonify(ret)
    response.status_code = 200
    return response


if __name__ == '__main__':
    p = argparse.ArgumentParser(prog="python backend_server.py", description="Handwriting App Backend")
    p.add_argument('-m', '--model', help="The model to be loaded",
                   type=str, default=default_model_name)
    args = p.parse_args()

    # Set device
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
    # Init models
    print('Loading the model...')
    g_models = [f for f in listdir(models_path) if
                isfile(join(models_path, f)) and f.endswith('.pt') and f[0] == 'G']
    g_path = ''
    for g in g_models:
        if args.model in g:
            g_path = join(models_path, g)
            break
    if g_path != '':
        g = torch.load(g_path)
        g.to(device=dev)
        g.eval()
        print('Loaded {}'.format(g_path))
    else:
        raise Exception('Could not find the model')

    print('Ready.')
    app.run()
