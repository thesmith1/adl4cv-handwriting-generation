import torch
from matplotlib.pyplot import imshow, show, figure
from numpy.random import randn
from torch.nn.modules.loss import _Loss, BCELoss
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, Pad
from pycrayon import CrayonClient
from componentsGAN import ConditionalGenerator, ConditionalDiscriminator
from condition_encoding import character_to_one_hot
from data_management.character_dataset import CharacterDataset
from global_vars import NOISE_LENGTH
from models import G1, D1, ConditionalDCGANDiscriminator, ConditionalDCGANGenerator
import datetime
import time

cc = CrayonClient('localhost')
cc.remove_all_experiments()
exp = cc.create_experiment('GAN Training - {}'.format(datetime.datetime.now()))


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
        z = torch.from_numpy(randn(1, NOISE_LENGTH)).to(device)
        output = self._G(z, c)
        if do_print:
            fig = figure()
            imshow(output[0].cpu().detach().numpy().squeeze(), cmap='Greys_r')
            fig.text(.5, 0.01, character)
            show()
        self._G.train()
        return output

    def train(self, n_epochs: int):
        max_GPU_memory = 0
        # Iterate epochs
        print('Starting epochs, GPU memory in use '
              'before loading the inputs: {} MB'.format(torch.cuda.memory_allocated(torch.cuda.current_device())/1e6))
        G_traced = torch.jit.trace(self._G.forward,
                                   (torch.randn(128, NOISE_LENGTH).to(device=device), torch.randn(128, 70)))
        D_traced = torch.jit.trace(self._D.forward,
                                   (torch.randn(128, 1, 64, 64).to(device=device), torch.randn(128, 70)))
        for epoch in range(1, n_epochs + 1):
            start_time = time.time()
            # Iterate the dataset
            for batch_count, (X, c, style) in enumerate(self._dataset_loader):  # TODO: use style
                # Reset gradient
                self._D_optim.zero_grad()

                # Sample data
                z = torch.from_numpy(randn(len(X), NOISE_LENGTH)).to(device=device)
                X = X.to(device=device)
                c = character_to_one_hot(c)
                c = torch.from_numpy(c).to(device=device)

                # Discriminator forward-loss-backward-update
                # G_sample = self._G(z, c)
                G_sample = G_traced(z, c)
                # D_real = self._D(X, c)
                D_real = D_traced(X, c)
                # D_fake = self._D(G_sample.detach(), c)
                D_fake = D_traced(G_sample.detach(), c)

                zero_label = torch.zeros(len(X)).to(device=device)
                one_label = torch.ones(len(X)).to(device=device)

                D_loss_real = self._D_loss(D_real, one_label)
                D_loss_fake = self._D_loss(D_fake, zero_label)
                D_loss = Variable(D_loss_real + D_loss_fake, requires_grad=True)

                D_loss.backward()
                self._D_optim.step(None)

                # Reset gradient
                self._G_optim.zero_grad()

                # Generator forward-loss-backward-update
                z = torch.from_numpy(randn(len(X), NOISE_LENGTH)).to(device)

                # G_sample = self._G(z, c)
                G_sample = G_traced(z, c)
                # D_fake = self._D(G_sample, c)
                D_fake = D_traced(G_sample, c)

                G_loss = Variable(self._G_loss(D_fake, one_label), requires_grad=True)

                G_loss.backward()
                self._G_optim.step(None)

                # Store max allocated GPU memory
                max_GPU_memory = max(max_GPU_memory, torch.cuda.max_memory_allocated(torch.cuda.current_device())/1e6)

                # Logging
                exp.add_scalar_value("Loss/Generator", G_loss.item())
                exp.add_scalar_value("Loss/Discriminator", D_loss.item())
                exp.add_scalar_value("Discriminator response/to real images (average)", D_real.mean().item())
                exp.add_scalar_value("Discriminator response/to fake images (average)", D_fake.mean().item())
                print('Epoch {}, batch: {}: D loss {:4f}, G loss {:4f}, '
                      'max GPU memory allocated: {:.2f} MB'.format(epoch, batch_count, D_loss, G_loss, max_GPU_memory),
                      end='\r')

            end_time = time.time()
            print('One epoch performed in {} s'.format(end_time - start_time))
            if epoch % 10 == 0:
                letter_to_generate = 'A' if randn(1) > 0 else 'O'
                self.generate(letter_to_generate, True)


if __name__ == '__main__':
    assert(torch.__version__ == '1.0.0')
    # Test cGAN class
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    g = ConditionalDCGANGenerator()
    d = ConditionalDCGANDiscriminator()
    g_adam = Adam(g.parameters(), lr=1e-3)
    d_adam = Adam(d.parameters(), lr=1e-5)
    transform = Compose([Pad(7), ToTensor()])
    dataset = CharacterDataset('../data/processed/', '../data/labels_test.txt', transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    gan = CGAN(g, d, BCELoss(), BCELoss(), g_adam, d_adam, loader)
    gan.train(1000)
    torch.save(g, "../data/models/gen.pth")
    torch.save(d, "../data/models/dis.pth")
