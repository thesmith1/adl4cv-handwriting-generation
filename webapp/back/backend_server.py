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

from utils.image_utils import generate_optimized_from_string, CONTRAST_INCREASE
from stitching import stitch

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

default_model_name = 'G_2019-01-22 22_33_20.613720_3600'
models_path = '../../data/models/'
assets_path = 'assets/'
encoding = 'utf-8'

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

Backend's local state:
- list of images of words (including prev and next spaces)
- image of current line (without black padding)
- string of transcription

Every request must contain the index of the character that is requested (the first)
'''


def ndarray_to_base64(arr: np.ndarray):
    with BytesIO() as output_bytes:
        arr = (arr / np.max(arr)) * 2 - 1
        image = Image.fromarray(skimage.img_as_ubyte(arr))
        image.save(output_bytes, 'JPEG')
        bytes_data = output_bytes.getvalue()
    return str(base64.b64encode(bytes_data), encoding)


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
    input_text = params['text']
    index = params['index']
    if index == len(state['text']):  # Insert at the end
        if not state['current_line']:  # New line
            input_text = ' ' + input_text
        characters = generate_optimized_from_string(g, input_text, params['style'], CONTRAST_INCREASE, device=dev)
        total = characters[0]
        for i in range(len(characters) - 1):
            total, _, _ = stitch(total, characters[i + 1])
        ret['current_line'] = total
        # Insert at the end, using current_line
        pass
    else:
        # Insert in the middle
        pass
    # Update state
    # state['current_line'] = ret['current_line']  # TODO: uncomment
    # TODO: Pad with black
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
