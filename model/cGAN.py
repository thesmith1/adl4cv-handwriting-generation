import torch
from numpy.random import randn
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
                 device: torch.device):
        self._G = G
        self._D = D
        self._G_optim = G_optim
        self._D_optim = D_optim
        self._G_loss = G_loss
        self._D_loss = D_loss
        self._device = device
        self._dataset_loader = dataset_loader
        self._G.to(device=self._device)
        self._D.to(device=self._device)

    def train(self, n_epochs: int):

        # Iterate epochs
        print('Starting epochs, GPU memory in use '
              'before loading the inputs: {} MB'.format(torch.cuda.memory_allocated(torch.cuda.current_device())/1e6))
        for epoch in range(n_epochs):
            # Iterate the dataset
            D_loss, G_loss = None, None
            for X, c, style in self._dataset_loader:  # TODO: use style

                # Reset gradient
                self._D_optim.zero_grad()

                # Sample data
                z = torch.from_numpy(randn(len(X), NOISE_LENGTH, 1)).to(device=device)
                X = X.to(device=device)
                c = character_to_one_hot(c)
                c = torch.from_numpy(c).to(device=device)

                # Discriminator forward-loss-backward-update
                G_sample = self._G(z, c)
                D_real = self._D(X, c)
                D_fake = self._D(G_sample, c)

                zero_label = torch.zeros(len(X), 1).to(device=device)
                one_label = torch.ones(len(X), 1).to(device=device)

                D_loss_real = self._D_loss(D_real, one_label)
                D_loss_fake = self._D_loss(D_fake, zero_label)
                D_loss = D_loss_real + D_loss_fake

                D_loss.backward()
                self._D_optim.step(None)

                # Generator forward-loss-backward-update

                # Reset gradient
                self._G_optim.zero_grad()
                z = torch.from_numpy(randn(len(X), NOISE_LENGTH, 1)).to(device)

                G_sample = self._G(z, c)
                D_fake = self._D(G_sample, c)

                G_loss = self._G_loss(D_fake, one_label)

                G_loss.backward()
                self._G_optim.step(None)

            if epoch % 5 == 0:
                print('Epoch {}: D loss {}, G loss {}'.format(epoch, D_loss, G_loss))


if __name__ == '__main__':
    # Test cGAN class
    use_cuda = True
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    g = G1()
    d = D1()
    g_adam = Adam(g.parameters())
    d_adam = Adam(d.parameters())
    dataset = CharacterDataset('../data/processed/', '../data/labels_test.txt', Compose([Pad(7), ToTensor()]))
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    gan = CGAN(g, d, BCELoss(), BCELoss(), g_adam, d_adam, loader, device)
    gan.train(100)
