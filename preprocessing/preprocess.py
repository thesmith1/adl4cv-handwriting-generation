import os
import cv2
import numpy as np
from utils.image_utils import is_image

base_dir = "../data"
source_dir = os.path.join(base_dir, "raw")
dest_dir = os.path.join(base_dir, "processed")

WHITE_THRESHOLD = 120
WHITE = 255
BORDER_WIDTH = 5


def process_image(image):
    # gray-scaling
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # median filter
    image = cv2.medianBlur(image, 3)

    # 2-means clustering white saturation
    mu = [0, 255]
    z = np.zeros_like(image)
    z_previous = None
    while np.any(z != z_previous):

        z_previous = z
        # E-step
        for i, row in enumerate(image):
            for j, element in enumerate(row):
                distances = [np.linalg.norm(element - mu_i) for mu_i in mu]
                z[i, j] = np.argmin(distances)
        # M-step
        mu = [np.mean(image[z == i]) for i in [0, 1]]

    image[z == 1] = WHITE

    # border removal
    image[0:BORDER_WIDTH, :] = WHITE
    image[-BORDER_WIDTH:, :] = WHITE
    image[:, 0:BORDER_WIDTH] = WHITE
    image[:, -BORDER_WIDTH:] = WHITE

    return image


if __name__ == '__main__':

    for filename in os.listdir(source_dir):
        if is_image(filename):
            print("Processing '%s'..." % filename)
            image_path = os.path.join(source_dir, filename)
            image = cv2.imread(image_path)
            processed_image = process_image(image)
            cv2.imwrite(os.path.join(dest_dir, filename), processed_image)
