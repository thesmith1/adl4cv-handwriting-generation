import os
import cv2
import sys
import numpy as np

lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)
ext_lib_path = os.path.abspath(os.path.join(__file__, '../../utils'))
sys.path.append(ext_lib_path)

from utils.global_vars import IMAGE_WIDTH, IMAGE_HEIGHT, INF_REMOVE_WIDTH, SUP_REMOVE_WIDTH
from utils.image_utils import is_image

base_dir = "../data/big/"
source_dir = os.path.join(base_dir, "raw")
dest_dir = os.path.join(base_dir, "processed")

DEBUG = False
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
BLACK = (0, 0, 0)
NOISE_REMOVAL_STRENGTH = 5


def color_k_means(image, mu_init, mask=None):

    orig_mask, orig_shape = mask, image.shape
    if mask is None:
        mask = np.ones(image.shape[:2], dtype=bool)
    image = image[mask]

    mu = np.array(mu_init)
    z = np.zeros_like(image, dtype=np.uint8)
    z_previous = np.ones_like(z)
    it = 0
    while not np.all(z == z_previous) and it < 100:

        z_previous = np.copy(z)

        # E-step
        distances = np.linalg.norm(image[None, :] - mu[:, None, :], axis=2)
        z = np.argmin(distances, axis=0)

        # M-step
        for i in range(mu.shape[0]):
            if np.sum(z == i) == 0:
                mu[i] = mu_init[i]
            else:
                mu[i] = np.mean(image[z == i])
        it += 1

    if orig_mask is None:
        z = z.reshape(orig_shape[:2])
    return z, mu


def process_image(image):

    if DEBUG:
        cv2.imshow("original", image)
        cv2.moveWindow("original", 0, 100)

    # extract a-channel
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image_lab)
    if DEBUG:
        cv2.imshow("a-channel", a)
        cv2.moveWindow("a-channel", 800, 100)

    # increase contrast
    image = np.clip((a-np.min(a))*1.5, 0, 255).astype(np.uint8)
    if DEBUG:
        cv2.imshow("contrast in lab", image)
        cv2.moveWindow("contrast in lab", 1200, 100)

    # noise removal
    image[image < 25] = 0
    image = cv2.medianBlur(image, ksize=NOISE_REMOVAL_STRENGTH)
    if DEBUG:
        cv2.imshow("noise removed", image)
        cv2.moveWindow("noise removed", 1600, 100)

    # final saturation
    image = np.clip(image * 10., 0., 255.).astype(np.uint8)

    # border removed
    image = image[SUP_REMOVE_WIDTH:-INF_REMOVE_WIDTH, :]

    # final resize
    # image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    if DEBUG:
        cv2.imshow("final result", image)
        cv2.moveWindow("final result", 2000, 100), cv2.waitKey(0)

    return image


if __name__ == '__main__':

    all_filenames = os.listdir(source_dir)
    for file_count, filename in enumerate(sorted(all_filenames, reverse=False)):
        if is_image(filename):
            print("Processing '%s' [%3d/%3d]..." % (filename, file_count + 1, len(all_filenames)), end='\r')
            image_path = os.path.join(source_dir, filename)
            img = cv2.imread(image_path)
            processed_image = process_image(img)
            cv2.imwrite(os.path.join(dest_dir, filename), processed_image)
