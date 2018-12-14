import cv2
import numpy as np
import os
from sys import exit
import argparse

FATAL = -1

valid_style_ids = [0, 1]
win_name = "Crop image"
dest_save_folder = "../data/big/raw"
rectangle_shape = (90, 30)  # height must be tuned to match the height of two lines in a college-ruled notebook
resizable = False
rectangle_color = (0, 0, 0)
DEBUG = False

# keys
QUIT_KEY = ord('q')
INPUT_KEY = ord('i')
LEFT_ARROW_KEY = 81
UP_ARROW_KEY = 82
RIGHT_ARROW_KEY = 83
DOWN_ARROW_KEY = 84
RETURN_KEY = 13
KEY_SPEED = 1


class Rectangle:

    def __init__(self, p0, width, height):
        self.x0, self.y0 = p0
        self.w, self.h = width, height

    def is_point_inside(self, point):
        x, y = point
        return self.x0 <= x <= (self.x0 + self.w) and self.y0 <= y <= (self.y0 + self.h)


class WindowOperationHandler:

    def __init__(self, window, image, init_rectangle, save_folder, text):
        self.window = window
        self.image = image
        self.image_drawable = np.copy(image)
        self.rectangle = init_rectangle
        self.save_folder = save_folder

        self.text = text
        self.previous_char = " "
        self.char_annotations = dict()

        self.taken_filenames = set()
        self.current_filename_index = self.load_current_index()

        self.dragging = False
        self.resizing = False
        self.previous_top_left_coordinates = (0, 0)
        self.previous_coordinates = (0, 0)

    def load_current_index(self):
        all_image_filenames = [filename for filename in os.listdir(self.save_folder) if ".jpg" in filename]
        all_indices = {int(filename.replace(".jpg", "")) for filename in all_image_filenames}
        current_index = max(all_indices) if len(all_indices) > 0 else 0
        return current_index

    def save_crop(self):

        # obtain crop
        x1, y1 = self.rectangle.x0, self.rectangle.y0
        x2, y2 = x1 + self.rectangle.w, y1 + self.rectangle.h
        cropped_image = self.image[y1:y2, x1:x2]

        # obtain image filename
        self.current_filename_index += 1
        filename = "{}.jpg".format(self.current_filename_index)

        # save image
        cv2.imwrite(os.path.join(self.save_folder, filename), cropped_image)

        # produce char annotation
        current_char = self.text[0]
        next_char = self.text[1]
        print("Crop saved as '%s' in '%s'. "
              "Char annotation: prev='%s', curr='%s', next='%s'." % (filename, self.save_folder, self.previous_char,
                                                                     current_char, next_char))
        self.text = self.text[1:]  # slide on text by one
        char_annotation = map(lambda char: "_" if char == " " else char, (self.previous_char, current_char, next_char))
        self.char_annotations[filename] = char_annotation
        self.previous_char = current_char

    def mouse_callback(self, event, x, y, flags=None, param=None):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if self.rectangle.is_point_inside(point=(x, y)):
                self.save_crop()
                x1, y1 = self.rectangle.x0, self.rectangle.y0
                x2, y2 = self.rectangle.x0 + self.rectangle.w, self.rectangle.y0 + self.rectangle.h
                cv2.rectangle(self.image_drawable, pt1=(x1, y1), pt2=(x2, y2), color=rectangle_color, thickness=-1)
                if DEBUG:
                    print("Detected double click inside rectangle!")
            else:
                if DEBUG:
                    print("Detected double click outside rectangle!")
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.rectangle.is_point_inside(point=(x, y)):
                self.dragging = True
                self.previous_top_left_coordinates = self.rectangle.x0, self.rectangle.y0
                self.previous_coordinates = x, y
                if DEBUG:
                    print("Dragging")
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.rectangle.x0 = x - self.previous_coordinates[0] + self.previous_top_left_coordinates[0]
                self.rectangle.y0 = y - self.previous_coordinates[1] + self.previous_top_left_coordinates[1]
                x1, y1 = self.rectangle.x0, self.rectangle.y0
                x2, y2 = self.rectangle.x0 + self.rectangle.w, self.rectangle.y0 + self.rectangle.h
                cv2.imshow("Zoom", self.image[y1:y2, x1:x2])
            elif self.resizing:
                self.rectangle.w = x - self.rectangle.x0
                self.rectangle.h = y - self.rectangle.y0
                if DEBUG:
                    print("Resizing")
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            if DEBUG:
                print("Stopped dragging")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if resizable and self.rectangle.is_point_inside(point=(x, y)):
                self.resizing = True
                if DEBUG:
                    print("Detected right click inside rectangle!")
        elif event == cv2.EVENT_RBUTTONUP:
            if self.resizing:
                self.resizing = False
                if DEBUG:
                    print("Stopped resizing")

    def keyboard_callback(self, key):

        delta_x = 0
        delta_y = 0
        if key == UP_ARROW_KEY:
            delta_y = -1 * KEY_SPEED
        elif key == DOWN_ARROW_KEY:
            delta_y = 1 * KEY_SPEED
        elif key == LEFT_ARROW_KEY:
            delta_x = -1 * KEY_SPEED
        elif key == RIGHT_ARROW_KEY:
            delta_x = 1 * KEY_SPEED
        elif key == RETURN_KEY:
            self.save_crop()
            x1, y1 = self.rectangle.x0, self.rectangle.y0
            x2, y2 = x1 + self.rectangle.w, y1 + self.rectangle.h
            cv2.rectangle(self.image_drawable, pt1=(x1, y1), pt2=(x2, y2), color=rectangle_color, thickness=-1)

        # apply delta
        if self.rectangle.x0 + delta_x >= 0 and self.rectangle.x0 + self.rectangle.w + delta_x < self.image.shape[1]:
            self.rectangle.x0 += delta_x
        if self.rectangle.y0 + delta_y >= 0 and self.rectangle.y0 + self.rectangle.h + delta_y < self.image.shape[0]:
            self.rectangle.y0 += delta_y

        x1, y1 = self.rectangle.x0, self.rectangle.y0
        x2, y2 = x1 + self.rectangle.w, y1 + self.rectangle.h
        cv2.imshow("Zoom", self.image[y1:y2, x1:x2])


def save_label_file(label_file_path, annotations, style_id):
    with open(label_file_path, "a") as lf:
        for filename, char_annotation in annotations.items():
            lf.write("{} {} {} {} {}\n".format(filename, *char_annotation, style_id))


def main_program_window(image, text, args):
    h, w, _ = image.shape
    cv2.namedWindow(win_name, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(win_name, image)

    h_rect, w_rect = rectangle_shape
    x0_rect, y0_rect = (w - w_rect) // 2, (h - h_rect) // 2
    rect = Rectangle(p0=(x0_rect, y0_rect), height=h_rect, width=w_rect)
    handler = WindowOperationHandler(win_name, image, rect, dest_save_folder, text)
    cv2.setMouseCallback(win_name, handler.mouse_callback)

    done = False
    while not done:

        # show image and crop rectangle
        image_drawable = np.copy(handler.image_drawable)
        cv2.rectangle(image_drawable, pt1=(rect.x0, rect.y0), pt2=(rect.x0 + rect.w, rect.y0 + rect.h),
                      color=rectangle_color, thickness=2)
        cv2.imshow(win_name, image_drawable)

        # check for key presses
        key = cv2.waitKey(1)
        if key == QUIT_KEY:  # quitting procedure

            valid = False
            save_response = None
            while not valid:
                save_response = input("You want to save? [y/n]")
                if save_response in ("y", "n"):
                    valid = True
                else:
                    print("Invalid response.")
            if save_response == "y":

                # check style was provided
                if args.style is None:
                    valid = False
                    while not valid:
                        insertion = input("No style id was provided, but it is required in order to save. Style id:")
                        try:
                            style_id = int(insertion)
                            if style_id not in valid_style_ids:
                                print("Invalid style id.")
                            else:
                                args.style = style_id
                                valid = True
                        except ValueError:
                            print("Invalid style id.")

                # save label file
                print("Saving label file...", end='')
                save_label_file(args.output_label_file, handler.char_annotations, args.style)
                print("done.")

            done = True

        elif key == INPUT_KEY:
            insertion = input('Append string in text buffer:')
            text.extend(insertion)
        elif key != -1:
            handler.keyboard_callback(key)


def main(args):
    # open image
    image_path = args.image_file
    img = cv2.imread(image_path)
    if img is None:
        print("Invalid path to image specified. '%s' not found" % image_path)
        exit(FATAL)

    # load text
    with open(args.input_text_file, "r") as f:
        text = f.read()

    # check label file
    assert os.path.isfile(args.output_label_file), "Invalid label file."

    main_program_window(img, text, args)


if __name__ == '__main__':
    # arg parse
    p = argparse.ArgumentParser(prog="python crop_utility.py", description="To produce annotated crops")
    p.add_argument('input_text_file', help="Requires to specify an input text file for labeling",
                   type=str)
    p.add_argument('output_label_file', help="Requires to specify an output label file",
                   type=str)
    p.add_argument('image_file', help="Requires to specify the path of the image to crop", type=str)
    p.add_argument('-s', '--style', help="Allows to specify the style id for the characters inside the image",
                   type=int, default=None, choices=valid_style_ids)
    arguments = p.parse_args()

    main(arguments)
