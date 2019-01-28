import argparse
import os
import sys
from os import listdir
from os.path import join, isfile

import numpy as np
import torch
from matplotlib.pyplot import imshow, show, figure, plot
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage

lib_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.append(lib_path)

from utils.global_vars import IMAGE_WIDTH, rectangle_shape, SUP_REMOVE_WIDTH, INF_REMOVE_WIDTH
from utils.image_utils import generate, generate_optimized, MEAN_OF_THREE, CONTRAST_INCREASE, THRESHOLD_SATURATE

height_dim = 0
width_dim = 1
OVERLAP_LIMIT_THRESHOLD = 0.95
BLACK_CORRELATION_OFFSET = 0.3
WINDOW_RAMP_PROPORTION = 0.25


# Accepts ndarrays or torch.Tensor of ndims=2
def stitch(t1, t2):
    if type(t1) == torch.Tensor:
        t1 = t1.numpy()
    if type(t2) == torch.Tensor:
        t2 = t2.numpy()
    # Offset to include correlation on the black pixels
    t1 = t1 - BLACK_CORRELATION_OFFSET
    t2 = t2 - BLACK_CORRELATION_OFFSET
    overlap_limit = int(t2.shape[width_dim] * OVERLAP_LIMIT_THRESHOLD)
    ramp_length = int((overlap_limit - 1) * WINDOW_RAMP_PROPORTION)
    corr_vals = []
    window = np.ones((overlap_limit - 1, ))
    window[:ramp_length] = np.linspace(0.3, 1, ramp_length)
    window[-ramp_length:] = np.linspace(1, 0.3, ramp_length)
    for overlap_idx in range(1, overlap_limit):
        normalization_coeff = t1[:, -overlap_idx:].shape[0] * t1[:, -overlap_idx:].shape[1]
        corr = np.sum(np.sum((t1[:, -overlap_idx:] * t2[:, :overlap_idx]))) / normalization_coeff
        corr_vals.append(corr)
    best_overlap_idx = np.argmax(corr_vals * window) + 1
    # Correct offset
    t1 = t1 + BLACK_CORRELATION_OFFSET
    t2 = t2 + BLACK_CORRELATION_OFFSET
    left = t1[:, :-best_overlap_idx]
    common_area = t1[:, -best_overlap_idx:] + t2[:, :best_overlap_idx] / 2
    right = t2[:, best_overlap_idx:]
    return np.concatenate([left, common_area, right], axis=1), corr_vals, corr_vals * window


if __name__ == '__main__':
    dataset_path = '../data/big/processed/'
    labels_file = '../data/big/labels.txt'
    models_path = '../data/models/'

    p = argparse.ArgumentParser(prog="python train.py", description="Train GAN for handwriting generation")
    p.add_argument('-m', '--model', help="The model to be loaded",
                   type=str, required=True)
    p.add_argument('-s', '--style', help="The style to use",
                   type=int, required=True)
    args = p.parse_args()

    style = args.style

    final_image_height = (rectangle_shape[0] - SUP_REMOVE_WIDTH - INF_REMOVE_WIDTH) * IMAGE_WIDTH // rectangle_shape[1]
    finalizing_transform = Compose([ToPILImage(), Resize((final_image_height, IMAGE_WIDTH)), ToTensor()])

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

    # c1 = 'H'
    # c2 = 'e'
    # c3 = 'l'
    # c4 = 'l'
    # s1 = generate_optimized(g, (c1, c2, c3), style, CONTRAST_INCREASE, dev)
    # s2 = generate_optimized(g, (c2, c3, c4), style, CONTRAST_INCREASE, dev)
    # res, corr, w_corr = stitch(s1, s2)
    # imshow(res, cmap='Greys_r')
    # show()
    # plot(corr)
    # show()
    # plot(w_corr)
    # show()

    while True:
        print('Enter some text:')
        text = input()
        if text == 'quit':
            break
        text = ' ' + text + ' '
        characters = [generate_optimized(g, (text[i], text[i + 1], text[i + 2]), style, CONTRAST_INCREASE, dev) for i in
                      range(len(text) - 2)]
        total = characters[0]
        for i in range(len(characters) - 1):
            total, _, _ = stitch(total, characters[i + 1])

        f = figure()
        imshow(total, cmap='Greys_r')
        show()
