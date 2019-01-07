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
            nn.ReLU(),
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
            nn.ConvTranspose2d(IMAGE_WIDTH * 16, IMAGE_WIDTH * 8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(IMAGE_WIDTH * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(IMAGE_WIDTH * 8, IMAGE_WIDTH * 4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(IMAGE_WIDTH * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(IMAGE_WIDTH * 4, IMAGE_WIDTH * 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(IMAGE_WIDTH * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(IMAGE_WIDTH * 2, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z: Variable, c: Variable):
        latent_input = torch.cat([z, c], dim=1).view(-1, NOISE_LENGTH + NUM_CHARS).float().to(device=self._device)
        input_layers = self._project(latent_input).view(-1, IMAGE_WIDTH * 16, 4, 4)
        return self._main(input_layers)

    def to(self, *args, **kwargs):
        self._device = kwargs.get('device')
        self._project.to(device=self._device)
        self._main.to(device=self._device)


class ConditionalDCGANDiscriminator(ConditionalDiscriminator):
    """
    Conditional Deep Convolutional GAN Discriminator from author of DCGAN paper
    Link to existing implementation: https://github.com/Newmu/dcgan_code/blob/master/mnist/train_cond_dcgan.py
    """
    def __init__(self):
        super().__init__()
        self._device = None
        self._reshape_condition = Reshape((-1, NUM_CHARS, 1, 1))
        self._conv1 = nn.Sequential(
            nn.Conv2d(1 + NUM_CHARS, 1 + NUM_CHARS, 5, 2, 2),
            nn.LeakyReLU()
        )
        self._conv2 = nn.Sequential(
            nn.Conv2d(1 + NUM_CHARS * 2, IMAGE_WIDTH + NUM_CHARS, 5, 2, 2),
            nn.BatchNorm2d(IMAGE_WIDTH + NUM_CHARS),
            nn.LeakyReLU()
        )
        self._flatten = Reshape((-1, (IMAGE_WIDTH // 4) * (IMAGE_WIDTH // 4) * (IMAGE_WIDTH + NUM_CHARS)))
        self._linear1 = nn.Sequential(
            nn.Linear((IMAGE_WIDTH // 4) * (IMAGE_WIDTH // 4) * (IMAGE_WIDTH + NUM_CHARS) + NUM_CHARS, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU()
        )
        self._linear2 = nn.Sequential(
            nn.Linear(1024 + NUM_CHARS, 1),
            nn.Sigmoid()
        )

    def to(self, *args, **kwargs):
        self._device = kwargs.get('device')
        self._conv1.to(device=self._device)
        self._conv2.to(device=self._device)
        self._linear1.to(device=self._device)
        self._linear2.to(device=self._device)

    def conv_cond_concat(self, x, y):
        ones = torch.ones((x.shape[0], y.shape[1], x.shape[2], x.shape[3])).to(device=self._device)
        return torch.cat([x, y*ones], dim=1)

    def forward(self, x: Variable, c: Variable):
        x = x.float().to(device=self._device)
        c = c.float().to(device=self._device)
        c_reshaped = self._reshape_condition(c)
        x = self.conv_cond_concat(x, c_reshaped)
        x = self._conv1(x)
        x = self.conv_cond_concat(x, c_reshaped)
        x = self._conv2(x)
        x = self._flatten(x)
        x = torch.cat([x, c], dim=1)
        x = self._linear1(x)
        x = torch.cat([x, c], dim=1)
        x = self._linear2(x)
        return x.squeeze()


class ConditionalDCGANGenerator(ConditionalGenerator):
    """
    Conditional Deep Convolutional GAN Generator from author of DCGAN paper
    Link to existing implementation: https://github.com/Newmu/dcgan_code/blob/master/mnist/train_cond_dcgan.py
    """
    def __init__(self):
        super().__init__()
        self._device = None
        self._reshape_condition = Reshape((-1, NUM_CHARS, 1, 1))
        self._linear1 = nn.Sequential(
            nn.Linear(NOISE_LENGTH + NUM_CHARS, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        self._linear2 = nn.Sequential(
            nn.Linear(1024 + NUM_CHARS, IMAGE_WIDTH * 2 * (IMAGE_WIDTH // 4) * (IMAGE_WIDTH // 4)),
            nn.BatchNorm1d(IMAGE_WIDTH * 2 * (IMAGE_WIDTH // 4) * (IMAGE_WIDTH // 4)),
            nn.ReLU(True)
        )
        self._reshape_input = Reshape((-1, IMAGE_WIDTH * 2, IMAGE_WIDTH // 4, IMAGE_WIDTH // 4))
        self._conv1 = nn.Sequential(
            nn.ConvTranspose2d(IMAGE_WIDTH * 2 + NUM_CHARS, IMAGE_WIDTH, 5, 2, 2, 1),
            nn.BatchNorm2d(IMAGE_WIDTH),
            nn.ReLU(True)
        )
        self._conv2 = nn.Sequential(
            nn.ConvTranspose2d(IMAGE_WIDTH + NUM_CHARS, 1, 5, 2, 2, 1),
            nn.Sigmoid()
        )

    def to(self, *args, **kwargs):
        self._device = kwargs.get('device')
        self._linear1.to(device=self._device)
        self._linear2.to(device=self._device)
        self._conv1.to(device=self._device)
        self._conv2.to(device=self._device)

    def conv_cond_concat(self, x, y):
        ones = torch.ones((x.shape[0], y.shape[1], x.shape[2], x.shape[3])).to(device=self._device)
        return torch.cat([x, y*ones], dim=1)

    def forward(self, z: Variable, c: Variable):
        z = z.float().to(device=self._device)
        c = c.float().to(device=self._device)
        c_reshaped = self._reshape_condition(c)
        x = torch.cat([z, c], dim=1).view(-1, NOISE_LENGTH + NUM_CHARS)
        x = self._linear1(x)
        x = torch.cat([x, c], dim=1).float()
        x = self._linear2(x)
        x = self._reshape_input(x)
        x = self.conv_cond_concat(x, c_reshaped)
        x = self._conv1(x)
        x = self.conv_cond_concat(x, c_reshaped)
        x = self._conv2(x)
        return x
