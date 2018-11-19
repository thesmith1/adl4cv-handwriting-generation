from torch.autograd import Variable
import torch.nn as nn
import torch

from componentsGAN import ConditionalDiscriminator, ConditionalGenerator
from global_vars import NOISE_LENGTH, IMAGE_WIDTH


class D1(ConditionalDiscriminator):
    def __init__(self):
        super().__init__()
        self._main = nn.Sequential(
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
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(IMAGE_WIDTH * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Variable, c: Variable):
        self._main(x)


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
        self._main(latent_input)
