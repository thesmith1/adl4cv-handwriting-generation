import torch
from numpy.random import randn
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss, BCELoss
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Pad

from componentsGAN import ConditionalGenerator, ConditionalDiscriminator
from condition_encoding import character_to_one_hot
from data_management.character_dataset import CharacterDataset
from global_vars import NOISE_LENGTH
from models import G1, D1


class CGAN:
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
        if self._cuda:
            self._G.cuda()
            self._D.cuda()
        print(torch.cuda.memory_allocated(0))

    def train(self, n_epochs: int):
        zero_label = Variable(torch.zeros(loader.batch_size, 1))
        one_label = Variable(torch.ones(loader.batch_size, 1))
        if self._cuda:
            zero_label = zero_label.cuda()
            one_label = one_label.cuda()

        # Iterate epochs
        for epoch in range(n_epochs):
            # Iterate the dataset
            for X, c, style in self._dataset_loader:  # TODO: use style
                # Sample data
                z = Variable(torch.from_numpy(randn(loader.batch_size, NOISE_LENGTH, 1)))
                X = Variable(X)
                c = character_to_one_hot(c)
                c = Variable(torch.from_numpy(c))
                if self._cuda:
                    z = z.cuda()
                    c = c.cuda()
                    X = X.cuda()

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
                z = Variable(torch.from_numpy(randn(loader.batch_size, NOISE_LENGTH, 1)))
                if self._cuda:
                    z.cuda()
                G_sample = self._G(z, c)
                D_fake = self._D(G_sample, c)

                G_loss = self._G_loss(D_fake, one_label)

                G_loss.backward()
                self._G_optim.step(None)

                # Reset gradient
                self._G_optim.zero_grad()

                if epoch % 5 == 0:
                    print('Epoch {}: D loss {}, G loss {}'.format(epoch, D_loss, G_loss))


if __name__ == '__main__':
    # Test cGAN class
    use_cuda = True
    g = G1(use_cuda)
    d = D1(use_cuda)
    g_adam = Adam(g.parameters())
    d_adam = Adam(d.parameters())
    dataset = CharacterDataset('../data/img/', '../data/labels_test.txt', Compose([Pad(7), ToTensor()]))
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    gan = CGAN(g, d, BCELoss(), BCELoss(), g_adam, d_adam, loader, use_cuda)
    gan.train(100)
