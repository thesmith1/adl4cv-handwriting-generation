import os
import cv2
import numpy as np
from utils.image_utils import is_image

base_dir = "../data/big/"
source_dir = os.path.join(base_dir, "raw")
dest_dir = os.path.join(base_dir, "processed")

DEBUG = False
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
BLACK = (0, 0, 0)
BLACK_LINE_WIDTH = 20  # Prev 16
MORPH_ELEMENT_SHAPE = (1, 10)
BORDER_WIDTH = 25
NOISE_REMOVAL_STRENGTH = 7


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

    # 2-means clustering to find background
    mu = [GRAY, WHITE]
    z, _ = color_k_means(image, mu)
    if DEBUG:
        cv2.imshow("original", image)
    image[z == 1] = BLACK
    if DEBUG:
        cv2.imshow("background-clustered", image)

    # gray-scaling
    image[np.all(image != BLACK, axis=-1)] = WHITE
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        cv2.imshow("gray-scaled", image)

    # line removal with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_ELEMENT_SHAPE)
    h, w = image.shape
    image_part_with_line = image[(h - BLACK_LINE_WIDTH)//2:(h + BLACK_LINE_WIDTH)//2, :]

    cv2.erode(src=image_part_with_line, dst=image_part_with_line,
              kernel=kernel, borderValue=(-1, -1), iterations=1)
    if DEBUG:
        cv2.imshow("post erosion", image)

    cv2.dilate(src=image_part_with_line, dst=image_part_with_line,
               kernel=kernel, borderValue=(-1, -1), iterations=1)
    if DEBUG:
        cv2.imshow("post dilatation", image)

    # noise removal
    image = cv2.medianBlur(image, ksize=NOISE_REMOVAL_STRENGTH)
    if DEBUG:
        cv2.imshow("noise removed", image)

    # border removal
    image[:BORDER_WIDTH, :] = 0
    image[-BORDER_WIDTH:, :] = 0
    if DEBUG:
        cv2.imshow("final result", image), cv2.waitKey(0)

    return image


if __name__ == '__main__':

    all_filenames = os.listdir(source_dir)
    for file_count, filename in enumerate(sorted(all_filenames)):
        if is_image(filename):
            print("Processing '%s' [%3d/%3d]..." % (filename, file_count + 1, len(all_filenames)), end='\r')
            image_path = os.path.join(source_dir, filename)
            img = cv2.imread(image_path)
            processed_image = process_image(img)
            cv2.imwrite(os.path.join(dest_dir, filename), processed_image)
