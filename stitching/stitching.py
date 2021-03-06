import argparse
import os
import sys
from os import listdir
from os.path import join, isfile
import numpy as np
import cv2
import torch
from matplotlib.pyplot import imshow, show, figure, axis
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage

lib_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.append(lib_path)

from utils.global_vars import IMAGE_WIDTH, rectangle_shape, SUP_REMOVE_WIDTH, INF_REMOVE_WIDTH
from utils.image_utils import generate_optimized, CONTRAST_INCREASE

height_dim = 0
width_dim = 1
OVERLAP_LIMIT_THRESHOLD = 0.95
BLACK_CORRELATION_OFFSET = 0.5
WINDOW_RAMP_PROPORTION = 0.25
WINDOW_DAMP_FACTOR = 0.8
MAX_VERTICAL_T2_SHIFT = 8
overlap_limit = int(IMAGE_WIDTH * OVERLAP_LIMIT_THRESHOLD)
DEBUG = False


def shift_vertically(img, shift, pad_value=0.):
    if shift > 0:
        img = np.concatenate([img[shift:, :], pad_value*np.ones((shift, img.shape[width_dim]))], axis=0)
    elif shift < 0:
        img = np.concatenate([pad_value*np.ones((-shift, img.shape[width_dim])), img[:shift, :]], axis=0)
    return img


def stitch(t1, t2):
    """
    :param t1: ndarray|Tensor with number of dimensions = 2
    :param t2: ndarray|Tensor with number of dimensions = 2
    :return: ndarray containing t1, t2 stitched
    """
    # re-cast
    if type(t1) == torch.Tensor:
        t1 = t1.numpy()
    if type(t2) == torch.Tensor:
        t2 = t2.numpy()

    if DEBUG:
        imshow(t1, cmap='Greys_r')
        axis('off')
        figure()
        imshow(t2, cmap='Greys_r')
        axis('off')
        show()

    # Offset to include correlation on the black pixels
    t1 = t1 - BLACK_CORRELATION_OFFSET
    t2 = t2 - BLACK_CORRELATION_OFFSET

    # construct weight window
    ramp_length = int((overlap_limit - 1) * WINDOW_RAMP_PROPORTION)
    window = np.ones((overlap_limit - 1, ))
    window[:ramp_length] = np.linspace(WINDOW_DAMP_FACTOR, 1, ramp_length)
    window[-ramp_length:] = np.linspace(1, WINDOW_DAMP_FACTOR, ramp_length)

    # compute correlation
    corr_values = np.zeros((MAX_VERTICAL_T2_SHIFT * 2 + 1, overlap_limit,))
    for ver_shift in range(-MAX_VERTICAL_T2_SHIFT, MAX_VERTICAL_T2_SHIFT + 1):

        # produce shifted version of t2
        t2_shifted = shift_vertically(t2, ver_shift, pad_value=-BLACK_CORRELATION_OFFSET)

        for overlap_idx in range(1, overlap_limit):
            norm_coef = np.prod(t1[:, -overlap_idx:].shape)
            corr_values[ver_shift, overlap_idx] = np.sum(t1[:, -overlap_idx:] * t2_shifted[:, :overlap_idx]) / norm_coef

    best_overlap = np.argmax(corr_values[:, 1:] * window[None, :])
    best_overlap_ver, best_overlap_hor = np.unravel_index(best_overlap, dims=corr_values[:, 1:].shape)
    if DEBUG:
        print(corr_values[best_overlap_ver, best_overlap_hor])

    if best_overlap_ver > MAX_VERTICAL_T2_SHIFT:
        best_overlap_ver -= corr_values.shape[height_dim]
    best_overlap_hor += 1

    if DEBUG:
        print(best_overlap_ver, -best_overlap_hor)

    # insert vertical offset
    t2 = shift_vertically(t2, best_overlap_ver, pad_value=-BLACK_CORRELATION_OFFSET)

    # correct pixel value offset
    t1 = t1 + BLACK_CORRELATION_OFFSET
    t2 = t2 + BLACK_CORRELATION_OFFSET

    # stitch
    left = t1[:, :-best_overlap_hor]
    common_area = np.maximum(t1[:, -best_overlap_hor:], t2[:, :best_overlap_hor])
    common_area[common_area < 0.15] = 0
    common_area = cv2.medianBlur((common_area*255).astype('uint8'), 3).astype('float32')/255
    right = t2[:, best_overlap_hor:]
    return np.concatenate([left, common_area, right], axis=1), corr_values[1:], corr_values[:, 1:] * window[None, :]


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
    g_models = [f for f in listdir(models_path) if isfile(join(models_path, f)) and f.endswith('.pt') and f[0] == 'G']
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
        print('Enter some text:')
        text = input()
        if text == 'quit':
            break
        text = ' ' + text + ' '
        characters = [generate_optimized(g, (text[i], text[i + 1], text[i + 2]), style, CONTRAST_INCREASE, dev)
                      for i in range(len(text) - 2)]
        total = characters[0]
        for i in range(len(characters) - 1):
            total, _, _ = stitch(total, characters[i + 1])

        f = figure()
        imshow(total, cmap='Greys_r')
        axis('off')
        show()
