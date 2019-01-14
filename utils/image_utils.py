from torch import Tensor
from matplotlib.pyplot import figure, imshow

accepted_image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]


def produce_figure(img: Tensor, label: str):
    fig = figure()
    imshow(img.cpu().detach().numpy().squeeze(), cmap='Greys_r')
    fig.text(.5, 0.01, label, ha='center')
    return fig


def is_image(filename):
    return any(filename.endswith("." + image_extension) for image_extension in accepted_image_extensions)
