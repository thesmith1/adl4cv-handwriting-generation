import argparse
import os
import sys
from datetime import datetime
from os import listdir
from os.path import isfile, join

from tensorboardX import SummaryWriter

import matplotlib as mpl
mpl.use('Agg')  # Needed if running on Google Cloud
from matplotlib.pyplot import show
from numpy.random import randn
import torch
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor

lib_path = os.path.abspath(os.path.join(__file__, '.'))
sys.path.append(lib_path)

from utils.global_vars import *
from utils.condition_encoding import character_to_one_hot
from utils.image_utils import produce_figure
from model.componentsGAN import ConditionalGenerator

models_path = './data/models/'
logs_path = './model/runs/'
step = 0


def generate(generator: ConditionalGenerator, characters: tuple, style: int, transform,
             writer: SummaryWriter, device, do_print: bool = False):
    assert len(characters) == 3
    assert style in (0, 1)
    character_conditioning = character_to_one_hot(characters)
    character_conditioning = torch.from_numpy(character_conditioning)
    character_conditioning = torch.cat([character_conditioning, style * torch.ones((1, 1), dtype=torch.double)],
                                       dim=1).to(device=device)
    z = torch.from_numpy(randn(1, NOISE_LENGTH)).to(device)
    style_str = '_P'
    if style:
        style_str = '_G'
    image = generator(z, character_conditioning).cpu().detach().squeeze()
    image = transform(image.unsqueeze(0))
    if do_print:
        produce_figure(image, "prev: {}, curr: {}, next: {}, style: {}".format(*characters, style))
        show()
        writer.add_image('Generated/' + ''.join(characters) + style_str, image, global_step=step)


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
                isfile(join(models_path, f)) and f.endswith('.pth') and f[0] == 'G']
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

    print(join(logs_path, current_datetime))
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
            generate(g, tuple(list(key_input)), st, finalizing_transform, writer, dev, True)

        print("done.")
