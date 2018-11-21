import torch
from matplotlib.pyplot import imshow, show
from numpy.random import randn
from torch.nn.modules.loss import _Loss, BCELoss
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Pad

from componentsGAN import ConditionalGenerator, ConditionalDiscriminator
from condition_encoding import character_to_one_hot
from data_management.character_dataset import CharacterDataset
from global_vars import NOISE_LENGTH
from models import G1, D1, ConditionalDCGANDiscriminator, ConditionalDCGANGenerator


class CGAN:
    def __init__(self, G: ConditionalGenerator, D: ConditionalDiscriminator,
                 G_loss: _Loss, D_loss: _Loss,
                 G_optim: Optimizer, D_optim: Optimizer,
                 dataset_loader: DataLoader):
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

    def generate(self, character: str, do_print: bool = False):
        self._G.eval()
        assert len(character) == 1
        c = character_to_one_hot(character)
        c = torch.from_numpy(c).to(device=device)
        z = torch.from_numpy(randn(1, NOISE_LENGTH)).to(device=self._device)
        output = self._G(z, c)
        if do_print:
            imshow(output[0].cpu().detach().numpy().transpose(1, 2, 0).squeeze(), cmap='Greys_r')
            show()
        self._G.train()
        return output

    def train(self, n_epochs: int):
        max_GPU_memory = 0
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
                z = torch.from_numpy(randn(len(X), NOISE_LENGTH)).to(device=device)
                X = X.to(device=device)
                c = character_to_one_hot(c)
                c = torch.from_numpy(c).to(device=device)

                # Discriminator forward-loss-backward-update
                G_sample = self._G(z, c)
                D_real = self._D(X, c)
                D_fake = self._D(G_sample, c)

                zero_label = torch.zeros(len(X)).to(device=device)
                one_label = torch.ones(len(X)).to(device=device)

                D_loss_real = self._D_loss(D_real, one_label)
                D_loss_fake = self._D_loss(D_fake, zero_label)
                D_loss = D_loss_real + D_loss_fake

                D_loss.backward()
                self._D_optim.step(None)

                # Reset gradient
                self._G_optim.zero_grad()

                # Generator forward-loss-backward-update
                z = torch.from_numpy(randn(len(X), NOISE_LENGTH)).to(device)

                G_sample = self._G(z, c)
                D_fake = self._D(G_sample, c)

                G_loss = self._G_loss(D_fake, one_label)

                G_loss.backward()
                self._G_optim.step(None)

                # Store max allocated GPU memory
                max_GPU_memory = max(max_GPU_memory, torch.cuda.max_memory_allocated(torch.cuda.current_device())/1e6)

            if epoch % 10 == 0:
                print('Epoch {}: D loss {}, G loss {}'.format(epoch, D_loss, G_loss))
                print('Max GPU memory allocated: {} MB'.format(max_GPU_memory))
                self.generate('A', True)


if __name__ == '__main__':
    # Test cGAN class
    use_cuda = True
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    g = ConditionalDCGANGenerator()
    d = ConditionalDCGANDiscriminator()
    g_adam = Adam(g.parameters())
    d_adam = Adam(d.parameters(), lr=1e-4)
    transform = Compose([Pad(7), ToTensor()])
    dataset = CharacterDataset('../data/processed/', '../data/labels_test.txt', transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    gan = CGAN(g, d, BCELoss(), BCELoss(), g_adam, d_adam, loader)
    gan.train(1000)
