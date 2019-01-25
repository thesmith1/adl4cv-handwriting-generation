import datetime
import time

import torch
from numpy.random import randn
from pycrayon import CrayonClient
from torch.optim import Optimizer, Adam, RMSprop
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize

from componentsGAN import ConditionalGenerator, ConditionalDiscriminator
from condition_encoding import character_to_one_hot
from data_management.character_dataset import CharacterDataset
from global_vars import NOISE_LENGTH, NUM_CHARS, character_to_index_mapping
from models import ConditionalDCGANGenerator, ConditionalDCGANDiscriminator

cc = CrayonClient('localhost')
cc.remove_all_experiments()
exp = cc.create_experiment('wGAN Training - {}'.format(datetime.datetime.now()))

clamp_limit = 1e-2


class WCGAN:
    def __init__(self, G: ConditionalGenerator, D: ConditionalDiscriminator,
                 G_optim: Optimizer, D_optim: Optimizer,
                 dataset_loader: DataLoader,
                 dataset: CharacterDataset,
                 device: torch.device):
        self._G = G
        self._D = D
        self._G_optim = G_optim
        self._D_optim = D_optim
        self._device = device
        self._dataset_loader = dataset_loader
        self._dataset = dataset
        self._G.to(device=self._device)
        self._D.to(device=self._device)

    def train(self, num_epochs: int):
        current_character_index = 27  # 0 is already present
        max_GPU_memory = 0
        print('Starting epochs, GPU memory in use '
              'before loading the inputs: {} MB'.format(torch.cuda.memory_allocated(torch.cuda.current_device()) / 1e6))

        # produced JIT models
        G_traced = torch.jit.trace(self._G, (torch.randn(128, NOISE_LENGTH).to(self._device),
                                             torch.randn(128, NUM_CHARS).to(self._device)))
        D_traced = torch.jit.trace(self._D, (torch.randn(128, 1, 64, 64).to(self._device),
                                             torch.randn(128, NUM_CHARS).to(self._device)))

        # Epoch iteration
        for epoch in range(1, num_epochs + 1):
            if epoch % 20 == 0 and current_character_index < NUM_CHARS:
                self._dataset.add_character_to_training(
                    list(character_to_index_mapping.keys())[current_character_index])
                current_character_index = current_character_index + 1

            self._G.train()
            self._D.train()

            # Iterate over the dataset
            start_time = time.time()
            for batch_count, (X, c, style) in enumerate(self._dataset_loader):  # TODO: use style
                # Reset gradient
                self._D_optim.zero_grad()

                # Sample data
                z = torch.from_numpy(randn(len(X), NOISE_LENGTH)).to(device=self._device)
                X = X.to(device=self._device)
                c = character_to_one_hot(c)
                c = torch.from_numpy(c).to(device=self._device)

                # Discriminator forward-loss-backward-update
                G_sample = G_traced(z, c)
                D_real = D_traced(X, c)
                D_fake = D_traced(G_sample.detach(), c)

                D_loss = torch.mean(D_real) - torch.mean(D_fake)  # originally negative

                D_loss.backward()
                self._D_optim.step(None)

                # Weight clipping
                for p in self._D.parameters():
                    p.data.clamp_(-clamp_limit, clamp_limit)

                # Reset gradient
                self._G_optim.zero_grad()

                # Generator forward-loss-backward-update
                z = torch.from_numpy(randn(len(X), NOISE_LENGTH)).to(self._device)

                G_sample = G_traced(z, c)
                D_fake = D_traced(G_sample, c)

                G_loss = -torch.mean(D_fake)

                G_loss.backward()
                self._G_optim.step(None)

                # Store max allocated GPU memory
                max_GPU_memory = max(max_GPU_memory, torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1e6)

                # Logging
                exp.add_scalar_value("Loss/Generator", G_loss.item())
                exp.add_scalar_value("Loss/Discriminator", D_loss.item())
                exp.add_scalar_value("Discriminator response/to real images (average)", D_real.mean().item())
                exp.add_scalar_value("Discriminator response/to fake images (average)", D_fake.mean().item())
                print('Epoch {:2d}, batch {:2d}/{:2d}, D loss {:4f}, G loss {:4f}, '
                      'max GPU memory allocated {:.2f} MB'.format(epoch, batch_count + 1, len(self._dataset_loader),
                                                                  D_loss, G_loss, max_GPU_memory),
                      end='\r')

            end_time = time.time()
            print('\nEpoch completed in {:.2f} s'.format(end_time - start_time))
            # if epoch % 25 == 0:
            #     letter_to_generate = list(character_to_index_mapping.keys())[current_character_index - 1]
            #     self.generate(letter_to_generate, True)

            if epoch % 100 == 0:
                torch.save(self._G, "../data/models/wgen.pth")
                torch.save(self._D, "../data/models/wdis.pth")


if __name__ == '__main__':
    # Test wcGAN class
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    g = ConditionalDCGANGenerator()
    d = ConditionalDCGANDiscriminator()
    # g_optim = Adam(g.parameters(), lr=1e-4)
    # d_optim = Adam(d.parameters(), lr=1e-4)
    g_optim = RMSprop(g.parameters(), lr=1e-4)
    d_optim = RMSprop(d.parameters(), lr=1e-4)
    transform = Compose([Resize((64, 64)), ToTensor()])
    dataset = CharacterDataset('../data/big/processed/', '../data/labels/out_labels.txt', transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    gan = WCGAN(g, d, G_optim=g_optim, D_optim=d_optim, dataset_loader=loader, dataset=dataset,
               device=device)
    gan.train(10000)
