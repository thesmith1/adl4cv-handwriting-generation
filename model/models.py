from torch.autograd import Variable

from componentsGAN import ConditionalDiscriminator, ConditionalGenerator


class D1(ConditionalDiscriminator):
    def __init__(self):
        super().__init__()

    def forward(self, x: Variable, c: Variable):
        pass


class G1(ConditionalGenerator):
    def __init__(self):
        super().__init__()

    def forward(self, z: Variable, c: Variable):
        pass
