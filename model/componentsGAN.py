import abc
import os
import sys

from numpy.random import randn

import torch
from torch.autograd import Variable
from torch.nn import Module

lib_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.append(lib_path)
ext_lib_path = os.path.abspath(os.path.join(__file__, '../utils'))
sys.path.append(ext_lib_path)

from utils.condition_encoding import character_to_one_hot
from utils.global_vars import NOISE_LENGTH


class ConditionalDiscriminator(Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        super().__init__()

    def __call__(self, x: Variable, c: Variable):
        return self.forward(x, c)

    @abc.abstractmethod
    def to(self, *args, **kwargs):
        raise NotImplementedError('Should have implemented this.')

    @abc.abstractmethod
    def forward(self, x: Variable, c: Variable):
        raise NotImplementedError('Should have implemented this.')


class ConditionalGenerator(Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        super().__init__()

    def __call__(self, z: Variable, c: Variable):
        return self.forward(z, c)

    @abc.abstractmethod
    def to(self, *args, **kwargs):
        raise NotImplementedError('Should have implemented this.')

    @abc.abstractmethod
    def forward(self, z: Variable, c: Variable):
        raise NotImplementedError('Should have implemented this.')

    def generate(self, characters: tuple, style: int):
        self.eval()
        assert len(characters) == 3 and style in (0, 1)
        character_conditioning = torch.from_numpy(character_to_one_hot(characters))
        character_conditioning = torch.cat([character_conditioning, style*torch.ones((1, 1), dtype=torch.double)],
                                           dim=1).to(device=self._device)
        z = torch.from_numpy(randn(1, NOISE_LENGTH)).to(self._device)
        output = self.forward(z, character_conditioning).cpu().detach().squeeze()
        self.train()
        return output
