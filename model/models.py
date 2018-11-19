from torch.autograd import Variable
import torch.nn as nn
import torch

from componentsGAN import ConditionalDiscriminator, ConditionalGenerator
from global_vars import NOISE_LENGTH, IMAGE_WIDTH


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        pass  # TODO: implement flatten with correct dimensions


class D1(ConditionalDiscriminator):
    """
    Discriminator of DCGAN, adapted to accept conditioning input
    """
    def __init__(self):
        super().__init__()
        self._main_conv = nn.Sequential(
            nn.Conv2d(3, IMAGE_WIDTH, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(IMAGE_WIDTH, IMAGE_WIDTH * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_WIDTH * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(IMAGE_WIDTH * 2, IMAGE_WIDTH * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_WIDTH * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(IMAGE_WIDTH * 4, IMAGE_WIDTH * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_WIDTH * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self._flatten = Flatten()
        self._main_linear = nn.Sequential(
            nn.Linear(),  # TODO: specify the correct dimensions
            nn.Sigmoid()
        )

    def forward(self, x: Variable, c: Variable):
        x = self._main_conv(x)
        x = self._flatten(x)
        conditioned_input = torch.cat([x, c])
        return self._main_linear(conditioned_input)


class G1(ConditionalGenerator):
    """
    Generator of DCGAN, adapted to accept conditioning input
    """
    def __init__(self):
        super().__init__()
        self._main = nn.Sequential(
            nn.ConvTranspose2d(NOISE_LENGTH + 3 * 70, IMAGE_WIDTH * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(IMAGE_WIDTH * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(IMAGE_WIDTH * 8, IMAGE_WIDTH * 6, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_WIDTH * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(IMAGE_WIDTH * 4, IMAGE_WIDTH * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_WIDTH * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(IMAGE_WIDTH * 2, IMAGE_WIDTH, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_WIDTH),
            nn.ReLU(True),
            nn.ConvTranspose2d(IMAGE_WIDTH, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z: Variable, c: Variable):
        print(z)
        print(c)
        print(z.shape)
        latent_input = torch.cat([z, c])
        return self._main(latent_input)
