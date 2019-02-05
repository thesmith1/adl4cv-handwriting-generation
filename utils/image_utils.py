import matplotlib as mpl
# mpl.use('Agg')  # Needed if running on Google Cloud

import torch
from numpy import clip
from utils.condition_encoding import character_to_one_hot
from utils.global_vars import NOISE_LENGTH, rectangle_shape, SUP_REMOVE_WIDTH, INF_REMOVE_WIDTH, IMAGE_WIDTH
from model.componentsGAN import ConditionalGenerator
from torch import Tensor
from matplotlib.pyplot import figure, imshow
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage

accepted_image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]
final_image_height = (rectangle_shape[0] - SUP_REMOVE_WIDTH - INF_REMOVE_WIDTH) * IMAGE_WIDTH // rectangle_shape[1]
finalizing_transform = Compose([ToPILImage(), Resize((final_image_height, IMAGE_WIDTH)), ToTensor()])

# Optimization modes
MEAN_OF_THREE = 0
CONTRAST_INCREASE = 1
CONTRAST_STRENGTH = 1.25


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


def generate_resize(model: ConditionalGenerator, characters: tuple, style: int, device=torch.device('cpu')):
    sample = generate(model, characters, style, device)
    return finalizing_transform(sample.unsqueeze(0))[0, :, :]


def generate_optimized(model: ConditionalGenerator, characters: tuple, style: int, mode: int, device=torch.device('cpu')):
    if mode == MEAN_OF_THREE:
        out1 = generate_resize(model, characters, style, device)
        out2 = generate_resize(model, characters, style, device)
        out3 = generate_resize(model, characters, style, device)
        final = (out1 + out2 + out3) / 3
        return final
    elif mode == CONTRAST_INCREASE:
        out1 = generate_resize(model, characters, style, device)
        out1[out1 < 0.3] = 0
        final = clip(out1*CONTRAST_STRENGTH, 0, 1)
        return final


def generate_optimized_from_string(model: ConditionalGenerator, text: str, style: int, mode: int, device=torch.device('cpu')):
    return [generate_optimized(model, (text[i], text[i + 1], text[i + 2]), style, mode, device) for i in range(len(text) - 2)]
