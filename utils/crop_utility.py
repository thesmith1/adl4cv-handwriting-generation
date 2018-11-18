import cv2
import numpy as np
from sys import argv, exit
FATAL = -1

win_name = "Crop image"
rectangle_shape = (50, 50)
white = (255, 255, 255)
DEBUG = False


class Rectangle:

    def __init__(self, p0, width, height):
        self.x0, self.y0 = p0
        self.w, self.h = width, height

    def is_point_inside(self, point):

        x, y = point
        return self.x0 <= x <= (self.x0 + self.w) and self.y0 <= y <= (self.y0 + self.h)


class RectangleDragHandler:

    def __init__(self, window, rectangle):
        self.window = window
        self.rectangle = rectangle
        self.dragging = False
        self.resizing = False
        self.previous_top_left_coordinates = (0, 0)
        self.previous_coordinates = (0, 0)

    def mouse_callback(self, event, x, y, flags=None, param=None):

        if event == cv2.EVENT_LBUTTONDBLCLK:
            if self.rectangle.is_point_inside(point=(x, y)):
                # TODO: insert code to save image
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
            if self.rectangle.is_point_inside(point=(x, y)):
                self.resizing = True
                if DEBUG:
                    print("Detected right click inside rectangle!")
        elif event == cv2.EVENT_RBUTTONUP:
            if self.resizing:
                self.resizing = False
                if DEBUG:
                    print("Stopped resizing")


def main_program_window(image):
    h, w, _ = img.shape

    cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(win_name, image)

    h_rect, w_rect = rectangle_shape
    x0_rect, y0_rect = (w - w_rect) // 2, (h - h_rect) // 2
    rect = Rectangle(p0=(x0_rect, y0_rect), height=h_rect, width=w_rect)
    handler = RectangleDragHandler(win_name, rect)
    cv2.setMouseCallback(win_name, handler.mouse_callback)

    done = False
    while not done:

        image_with_rect = np.copy(image)
        cv2.rectangle(image_with_rect, pt1=(rect.x0, rect.y0), pt2=(rect.x0 + rect.w, rect.y0 + rect.h), color=white)
        cv2.imshow(win_name, image_with_rect)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            done = True


if __name__ == '__main__':
    if len(argv) != 2:
        print("Invalid input arguments. Usage: python crop_utility.py <image-path>")
        exit(FATAL)

    # open image
    image_path = argv[1]
    img = cv2.imread(image_path)
    if img is None:
        print("Invalid path to image specified. '%s' not found" % image_path)
        exit(FATAL)

    main_program_window(img)
