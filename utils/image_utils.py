
accepted_image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]


def is_image(filename):
    return any(filename.endswith("." + image_extension) for image_extension in accepted_image_extensions)
