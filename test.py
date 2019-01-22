import argparse
import os
import sys
from datetime import datetime
from os import listdir
from os.path import isfile, join

import matplotlib as mpl
mpl.use('Agg')  # Needed if running on Google Cloud

from tensorboardX import SummaryWriter
from matplotlib.pyplot import show
import torch
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor

lib_path = os.path.abspath(os.path.join(__file__, '.'))
sys.path.append(lib_path)

from utils.global_vars import *
from utils.image_utils import generate, produce_figure


models_path = './data/models/'
logs_path = './model/runs/'
step = 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(prog="python train.py", description="Train GAN for handwriting generation")
    p.add_argument('-m', '--model', help="The model to be loaded",
                   type=str, required=True)
    p.add_argument('-s', '--style', help="The style to use (0 or 1; if empty, samples of both styles will be produced)",
                   type=int, default=None)
    args = p.parse_args()

    current_datetime = str(datetime.now())

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

    print("Results available in experiment %s" % join(logs_path, current_datetime))
    writer = SummaryWriter(join(logs_path, current_datetime))

    # Prepare style(s) to use
    styles_to_produce = [0, 1]
    if args.style:
        styles_to_produce = [args.style]

    final_image_height = (rectangle_shape[0] - SUP_REMOVE_WIDTH - INF_REMOVE_WIDTH) * IMAGE_WIDTH // rectangle_shape[1]
    finalizing_transform = Compose([ToPILImage(), Resize((final_image_height, IMAGE_WIDTH)), ToTensor()])

    while True:
        key_input = input('Enter sequence of chars as [prev_char curr_char next_char] without spaces:')
        if key_input == 'quit':
            exit(True)
        print('Generating "{}"...'.format(key_input), end='')

        step = step + 1
        for st in styles_to_produce:
            characters = tuple(key_input)
            style_str = '_P'
            if st:
                style_str = '_G'
            output = generate(g, characters, st, device=dev)
            image = finalizing_transform(output.unsqueeze(0))
            produce_figure(image, "prev: {}, curr: {}, next: {}, style: {}".format(*characters, st))
            show()
            writer.add_image('Generated/' + ''.join(characters) + style_str, image, global_step=step)
        print("done.")
