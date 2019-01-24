import argparse
from os import listdir
from os.path import join, isfile
from flask import Flask, request, json, jsonify
import torch

app = Flask(__name__)
default_model_name = 'G_2019-01-21 22:34:36.093017'
models_path = '../../data/models/'

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


@app.route('/')
def index():
    return 'Index Page!'


@app.route('/example_post', methods=['GET', 'POST'], strict_slashes=False)
def example_post():
    if request.method == 'POST':
        return 'Yes'
    else:
        return 'No'


@app.route('/insert', methods=['GET'], strict_slashes=False)
def insert():
    print(json.dumps(request.json))
    # TODO: From the string in the request, generate characters and stitch
    response = jsonify({'code': 200, 'value': 'Ciao'})
    # TODO: Fill response with Base64 image data
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
