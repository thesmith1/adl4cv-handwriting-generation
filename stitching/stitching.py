import argparse
import os
import sys
from os import listdir
from os.path import join, isfile

import numpy as np
import torch
from matplotlib.pyplot import imshow, show, figure
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage

lib_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.append(lib_path)

from utils.global_vars import IMAGE_WIDTH, rectangle_shape, SUP_REMOVE_WIDTH, INF_REMOVE_WIDTH

height_dim = 2
width_dim = 1
OVERLAP_LIMIT_THRESHOLD = 0.75
BLACK_CORRELATION_OFFSET = 0.3


def stitch(t1: torch.Tensor, t2: torch.Tensor):
    t1 = t1.numpy()
    t2 = t2.numpy()
    # Offset to include correlation on the black pixels
    t1 = t1 - BLACK_CORRELATION_OFFSET
    t2 = t2 - BLACK_CORRELATION_OFFSET
    overlap_limit = int(t1.shape[width_dim] * OVERLAP_LIMIT_THRESHOLD)
    corr_vals = []
    for overlap_idx in range(1, overlap_limit):
        normalization_coeff = t1[0, :, -overlap_idx:].shape[0] * t1[0, :, -overlap_idx:].shape[1]
        corr = np.sum(np.sum((t1[0, :, -overlap_idx:] * t2[0, :, :overlap_idx]))) / normalization_coeff
        corr_vals.append(corr)
    best_overlap_idx = np.argmax(corr_vals) + 1
    # Correct offset
    t1 = t1 + BLACK_CORRELATION_OFFSET
    t2 = t2 + BLACK_CORRELATION_OFFSET
    left = t1[0, :, :-best_overlap_idx]
    common_area = t1[0, :, -best_overlap_idx:] + t2[0, :, :best_overlap_idx] / 2
    right = t2[0, :, best_overlap_idx:]
    return np.concatenate([left, common_area, right], axis=1)


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

    while True:
        print('Enter a sequence of 4 characters:')
        char_seq = input()
        if char_seq == 'quit':
            break
        assert len(char_seq) == 4
        [prev_char, curr_char, next_char, last_char] = list(char_seq)

        sample1 = g.generate((prev_char, curr_char, next_char), style)
        sample2 = g.generate((curr_char, next_char, last_char), style)
        sample1 = finalizing_transform(sample1.unsqueeze(0))
        sample2 = finalizing_transform(sample2.unsqueeze(0))

        # Stitch
        res = stitch(sample1, sample2)

        f = figure()
        imshow(res, cmap='Greys_r')
        show()
