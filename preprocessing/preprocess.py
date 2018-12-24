import os
import cv2
import numpy as np
from utils.image_utils import is_image

base_dir = "../data/big/"
source_dir = os.path.join(base_dir, "raw")
dest_dir = os.path.join(base_dir, "processed")

DEBUG = False
WHITE = (255, 255, 255)
GRAY = (125, 125, 125)
BLACK = (0, 0, 0)
INIT_RED = (125, 125, 150)
BLACK_LINE_THRESHOLD = 200
BLACK_LINE_WIDTH = 25
BORDER_WIDTH = 25
NOISE_REMOVAL_STRENGTH = 7
RED_CHANNEL = 2


def color_k_means(image, mu_init, mask=None):

    orig_mask, orig_shape = mask, image.shape
    if mask is None:
        mask = np.ones(image.shape[:2], dtype=bool)
    image = image[mask]

    mu = np.array(mu_init)
    z = np.zeros_like(image, dtype=np.uint8)
    z_previous = np.ones_like(z)

    while not np.all(z == z_previous):

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

    if orig_mask is None:
        z = z.reshape(orig_shape[:2])
    return z


def process_image(image):

    # 2-means clustering to find background
    mu = [GRAY, WHITE]
    z = color_k_means(image, mu)
    if DEBUG:
        cv2.imshow("original", image)
    image[z == 1] = BLACK
    if DEBUG:
        cv2.imshow("clustered", image)

    # line detection
    h, w = image.shape[:2]
    rows_with_lines = []
    for row in range((h - BLACK_LINE_WIDTH)//2, (h + BLACK_LINE_WIDTH)//2):
        line_pixel_count = 0
        consecutive = 0
        max_consecutive = 0
        for col in range(w):
            if np.all(image[row, col] != BLACK):
                line_pixel_count += 1
                consecutive += 1
            else:
                if consecutive > max_consecutive:
                    max_consecutive = consecutive
                consecutive = 0
        max_consecutive = max(consecutive, max_consecutive)
        if line_pixel_count > w / 5 and max_consecutive > w / 30:
            rows_with_lines.append(row)

    if len(rows_with_lines) > 1:
        for row_with_lines in rows_with_lines:
            if row_with_lines + 1 not in rows_with_lines and \
                    row_with_lines - 1 not in rows_with_lines:
                rows_with_lines.remove(row_with_lines)
    if DEBUG:
        print(rows_with_lines)

    foreground_mean = image[z != 1].mean(axis=0)
    foreground_std = image[z != 1].std(axis=0)
    image_normalized = (image - foreground_mean)/foreground_std
    for row_with_lines in rows_with_lines:
        for col in range(w):
            pixel = image[row_with_lines, col]
            if np.any(pixel != BLACK):
                pixel_normalized = image_normalized[row_with_lines, col]
                if pixel_normalized[RED_CHANNEL] < foreground_mean[RED_CHANNEL]:
                    image[row_with_lines, col] = BLACK
    if DEBUG:
        cv2.imshow("line removed", image)

    # noise removal
    image = cv2.medianBlur(image, ksize=NOISE_REMOVAL_STRENGTH)
    if DEBUG:
        cv2.imshow("noise removed", image)

    # border removal
    image[:BORDER_WIDTH, :] = BLACK
    image[-BORDER_WIDTH:, :] = BLACK

    # gray-scaling
    image[np.all(image != BLACK, axis=-1)] = WHITE
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        cv2.imshow("final result", gray_image), cv2.waitKey(0)

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
