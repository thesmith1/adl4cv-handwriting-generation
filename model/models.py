import torch
import torch.nn as nn
from torch.autograd import Variable

from componentsGAN import ConditionalDiscriminator, ConditionalGenerator
from global_vars import NOISE_LENGTH, IMAGE_WIDTH, NUM_CHARS


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(-1, 400 * 4 * 4)


class D1(ConditionalDiscriminator):
    """
    Discriminator of DCGAN, adapted to accept conditioning input
    """
    def __init__(self):
        super().__init__()
        self._device = None
        self._main_conv = nn.Sequential(
            nn.Conv2d(1, IMAGE_WIDTH, 4, 2, 1, bias=False),
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
            nn.Linear(400 * 4 * 4 + NUM_CHARS, 400),
            nn.Linear(400, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Variable, c: Variable):
        x = self._main_conv(x.float())
        x = self._flatten(x)
        conditioned_input = torch.cat([x, c.view(-1, NUM_CHARS).float()], dim=1).to(device=self._device)
        return self._main_linear(conditioned_input)

    def to(self, *args, **kwargs):
        self._device = kwargs.get('device')
        self._main_conv.to(device=self._device)
        self._main_linear.to(device=self._device)


class G1(ConditionalGenerator):
    """
    Generator of DCGAN, adapted to accept conditioning input
    """
    def __init__(self):
        super().__init__()
        self._device = None
        self._main = nn.Sequential(
            nn.ConvTranspose2d(NOISE_LENGTH + NUM_CHARS, IMAGE_WIDTH * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(IMAGE_WIDTH * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(IMAGE_WIDTH * 8, IMAGE_WIDTH * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_WIDTH * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(IMAGE_WIDTH * 4, IMAGE_WIDTH * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_WIDTH * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(IMAGE_WIDTH * 2, IMAGE_WIDTH, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_WIDTH),
            nn.ReLU(True),
            nn.ConvTranspose2d(IMAGE_WIDTH, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z: Variable, c: Variable):
        latent_input = torch.cat([z, c], dim=1).view(-1, NOISE_LENGTH+NUM_CHARS, 1, 1).float().to(device=self._device)
        return self._main(latent_input)

    def to(self, *args, **kwargs):
        self._device = kwargs.get('device')
        self._main.to(device=self._device)
