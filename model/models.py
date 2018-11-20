import torch
import torch.nn as nn
from torch.autograd import Variable

from componentsGAN import ConditionalDiscriminator, ConditionalGenerator
from global_vars import NOISE_LENGTH, IMAGE_WIDTH, NUM_CHARS


class Reshape(nn.Module):
    def __init__(self, shape: tuple):
        super().__init__()
        self._shape = shape

    def forward(self, x):
        return x.view(self._shape)


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
        self._flatten = Reshape((-1, 512 * 4 * 4))
        self._main_linear = nn.Sequential(
            nn.Linear(512 * 4 * 4 + NUM_CHARS, 400),
            nn.Linear(400, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Variable, c: Variable):
        x = self._main_conv(x.float())
        x = self._flatten(x)
        conditioned_input = torch.cat([x, c.view(-1, NUM_CHARS).float()], dim=1).to(device=self._device)
        return self._main_linear(conditioned_input).squeeze()

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
        self._project = nn.Sequential(nn.Linear(NOISE_LENGTH + NUM_CHARS, IMAGE_WIDTH * 16 * 4 * 4), nn.ReLU())
        self._main = nn.Sequential(
            nn.ConvTranspose2d(IMAGE_WIDTH * 16, IMAGE_WIDTH * 8, kernel_size=4, stride=2, padding=1, bias=False),
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
            nn.ConvTranspose2d(IMAGE_WIDTH, 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z: Variable, c: Variable):
        latent_input = torch.cat([z, c], dim=1).view(-1, NOISE_LENGTH+NUM_CHARS).float().to(device=self._device)
        input_layers = self._project(latent_input).view(-1, IMAGE_WIDTH * 16, 4, 4)
        return self._main(input_layers)

    def to(self, *args, **kwargs):
        self._device = kwargs.get('device')
        self._project.to(device=self._device)
        self._main.to(device=self._device)


class ConditionalDCGANDiscriminator(ConditionalDiscriminator):
    def __init__(self):
        super().__init__()

    def to(self, *args, **kwargs):
        pass

    def forward(self, x: Variable, c: Variable):
        pass


class ConditionalDCGANGenerator(ConditionalGenerator):
    def __init__(self):
        super().__init__()

    def to(self, *args, **kwargs):
        pass

    def forward(self, z: Variable, c: Variable):
        pass
