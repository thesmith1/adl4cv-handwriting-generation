import torch
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss, BCELoss
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader

from componentsGAN import ConditionalGenerator, ConditionalDiscriminator
from models import G1, D1
from data_management.character_dataset import CharacterDataset


class cGAN:
    def __init__(self, G: ConditionalGenerator, D: ConditionalDiscriminator,
                 G_loss: _Loss, D_loss: _Loss,
                 G_optim: Optimizer, D_optim: Optimizer,
                 dataset_loader: DataLoader,
                 is_cuda: bool):
        self._G = G
        self._D = D
        self._G_optim = G_optim
        self._D_optim = D_optim
        self._G_loss = G_loss
        self._D_loss = D_loss
        self._cuda = is_cuda
        self._dataset_loader = dataset_loader
        self._noise_shape = [100, 100]  # TODO: set noise shape
        if self._cuda:
            self._G.cuda()
            self._D.cuda()

    def train(self, n_epochs: int):
        zero_label = Variable(torch.zeros(1, 1))
        one_label = Variable(torch.ones(1, 1))

        # Iterate epochs
        for epoch in range(n_epochs):
            # Iterate the dataset
            for X, c, style in self._dataset_loader:  # TODO: use style
                # Sample data
                z = Variable(torch.randn(self._noise_shape))
                X = Variable(torch.from_numpy(X))
                c = Variable(torch.from_numpy(c))
                if self._cuda:
                    z.cuda()
                    c.cuda()
                    X.cuda()

                # Discriminator forward-loss-backward-update
                G_sample = self._G(z, c)
                D_real = self._D(X, c)
                D_fake = self._D(G_sample, c)

                D_loss_real = self._D_loss(D_real, one_label)
                D_loss_fake = self._D_loss(D_fake, zero_label)
                D_loss = D_loss_real + D_loss_fake

                D_loss.backward()
                self._D_optim.step(None)

                # Reset gradient
                self._D_optim.zero_grad()

                # Generator forward-loss-backward-update
                z = Variable(torch.randn(self._noise_shape))
                if self._cuda:
                    z.cuda()
                G_sample = self._G(z, c)
                D_fake = self._D(G_sample, c)

                G_loss = self._G_loss(D_fake, one_label)

                G_loss.backward()
                self._G_optim.step(None)

                # Reset gradient
                self._G_optim.zero_grad()

                # TODO: Optionally print


if __name__ == '__main__':
    # Test cGAN class
    g = G1()
    d = D1()
    g_adam = Adam(g.parameters())
    d_adam = Adam(d.parameters())
    d = CharacterDataset('../data/img/', '../data/labels_test.txt')
    loader = DataLoader(d, batch_size=3, shuffle=True)
    gan = cGAN(g, d, BCELoss(), BCELoss(), g_adam, d_adam, loader, True)
    gan.train(100)
