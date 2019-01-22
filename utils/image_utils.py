import matplotlib as mpl
mpl.use('Agg')  # Needed if running on Google Cloud

import torch
from utils.condition_encoding import character_to_one_hot
from utils.global_vars import NOISE_LENGTH
from torch import Tensor
from matplotlib.pyplot import figure, imshow

accepted_image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]


def generate(model, characters: tuple, style: int, device=torch.device('cpu')):
    model.eval()
    assert len(characters) == 3 and style in (0, 1)
    character_conditioning = torch.from_numpy(character_to_one_hot(characters))
    character_conditioning = torch.cat([character_conditioning, style * torch.ones((1, 1), dtype=torch.double)],
                                       dim=1).to(device=device)
    z = torch.randn(1, NOISE_LENGTH).to(device)
    output = model(z, character_conditioning).cpu().detach().squeeze()
    return output


def produce_figure(img: Tensor, label: str):
    fig = figure()
    imshow(img.cpu().detach().numpy().squeeze(), cmap='Greys_r')
    fig.text(.5, 0.01, label, ha='center')
    return fig


def is_image(filename):
    return any(filename.endswith("." + image_extension) for image_extension in accepted_image_extensions)
