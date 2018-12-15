import os
import cv2
import numpy as np
from utils.image_utils import is_image

base_dir = "../data/big/"
source_dir = os.path.join(base_dir, "raw")
dest_dir = os.path.join(base_dir, "processed")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (125, 125, 125)
BLACK_LINE_THRESHOLD = 200
BLACK_LINE_LENGTH = 15
BORDER_WIDTH = 5


def color_k_means(image, mu_init, channels_to_look=None):

    if channels_to_look is None:
        channels_to_look = [0, 1, 2]

    mu = mu_init
    z = np.zeros(image.shape[:2], dtype=np.uint8)
    z_previous = np.ones_like(z)

    while not np.all(z == z_previous):

        z_previous = np.copy(z)
        # E-step
        for i, row in enumerate(image):
            for j, pixel in enumerate(row):
                distances = [np.linalg.norm(pixel[channels_to_look] - mu_i) for mu_i in mu]
                z[i, j] = np.argmin(distances)
        # M-step
        for i in (0, 1):
            if np.sum(z == i) == 0:
                mu[i] = mu_init[i]
            else:
                mu[i] = np.mean(image[z == i])

    return z


def process_image(image):

    # 2-means clustering to find background
    mu = np.array([BLACK, WHITE])
    z = color_k_means(image, mu)
    cv2.imshow("original", image)
    image[z == 1] = BLACK
    cv2.imshow("clustered", image)

    # detect line
    h, w = image.shape[:2]
    for i in range(h // 2 - 5, h // 2 + 5):
        for j in range(w):
            window = image[i-2:i+2, j]
            distance = np.mean(image[i, j]) - np.mean(window)
            if distance > 45:
                image[i, j] = BLACK

    image[image != BLACK] = 255
    cv2.imshow("line removed", image), cv2.waitKey(0)

    # border removal
    image[:BORDER_WIDTH, :] = BLACK
    image[-BORDER_WIDTH:, :] = BLACK

    # gray-scaling
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray_image


if __name__ == '__main__':

    all_filenames = os.listdir(source_dir)
    for file_count, filename in enumerate(sorted(all_filenames)):
        if is_image(filename):
            print("Processing '%s' [%3d/%3d]..." % (filename, file_count + 1, len(all_filenames)), end='\r')
            image_path = os.path.join(source_dir, filename)
            img = cv2.imread(image_path)
            processed_image = process_image(img)
            cv2.imwrite(os.path.join(dest_dir, filename), processed_image)
