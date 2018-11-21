import os
import cv2
import numpy as np
from utils.image_utils import is_image

base_dir = "../data"
source_dir = os.path.join(base_dir, "raw")
dest_dir = os.path.join(base_dir, "processed")

WHITE = 255
BORDER_WIDTH = 5


def process_image(image):

    # gray-scaling
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2-means clustering white saturation
    mu = [80, 160]
    z = np.zeros_like(image, dtype=np.uint8)
    z_previous = np.ones_like(image)
    while not np.all(z == z_previous):

        z_previous = np.copy(z)
        # E-step
        for i, row in enumerate(image):
            for j, element in enumerate(row):
                distances = [np.linalg.norm(element - mu_i) for mu_i in mu]
                z[i, j] = np.argmin(distances)
        # M-step
        for i in (0, 1):
            if np.sum(z == i) == 0:
                mu[i] = 80 * (i + 1)
            else:
                mu[i] = np.mean(image[z == i])
    image[z == 1] = WHITE

    # border removal
    image[0:BORDER_WIDTH, :] = WHITE
    image[-BORDER_WIDTH:, :] = WHITE
    image[:, 0:BORDER_WIDTH] = WHITE
    image[:, -BORDER_WIDTH:] = WHITE

    # inversion
    image = 255 - image

    return image


if __name__ == '__main__':

    all_filenames = os.listdir(source_dir)
    for file_count, filename in enumerate(all_filenames):
        if is_image(filename):
            print("Processing '%s' [%3d/%3d]..." % (filename, file_count + 1, len(all_filenames)), end='\r')
            image_path = os.path.join(source_dir, filename)
            img = cv2.imread(image_path)
            processed_image = process_image(img)
            cv2.imwrite(os.path.join(dest_dir, filename), processed_image)
