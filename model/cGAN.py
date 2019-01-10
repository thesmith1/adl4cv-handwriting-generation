import datetime
import time
import os
import sys

import torch
from matplotlib.pyplot import show
from numpy.random import randn, choice
from numpy import concatenate, inf
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor

lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)
ext_lib_path = os.path.abspath(os.path.join(__file__, '../../utils'))
sys.path.append(ext_lib_path)

from componentsGAN import ConditionalGenerator, ConditionalDiscriminator
from condition_encoding import character_to_one_hot
from data_management.character_dataset import CharacterDataset
from global_vars import *
from image_utils import produce_figure


class CGAN:
    def __init__(self, G: ConditionalGenerator, D: ConditionalDiscriminator,
                 G_loss: _Loss, D_loss: _Loss,
                 G_optim: Optimizer, D_optim: Optimizer,
                 dataset_loader: DataLoader,
                 dataset: CharacterDataset,
                 device: torch.device,
                 writer: SummaryWriter,
                 current_datetime: str = None):
        self._G = G
        self._D = D
        self._G_optim = G_optim
        self._D_optim = D_optim
        self._G_loss = G_loss
        self._D_loss = D_loss
        self._device = device
        self._dataset_loader = dataset_loader
        self._dataset = dataset
        self._G.to(device=self._device)
        self._D.to(device=self._device)
        self._writer = writer
        if current_datetime:
            self._current_datetime = current_datetime
        else:
            self._current_datetime = datetime.datetime.now()

    def generate(self, character: str, style: int, do_print: bool = False):
        self._G.eval()
        assert len(character) == 1
        assert style in (0, 1)
        c = character_to_one_hot(character)
        c = torch.from_numpy(c).to(device=self._device)
        c = torch.cat([c, style*torch.ones((1, 1), dtype=torch.double).to(self._device)], dim=1)
        z = torch.from_numpy(randn(1, NOISE_LENGTH)).to(self._device)
        output = self._G(z, c).cpu().detach().squeeze()
        if do_print:
            produce_figure(output, character)
            show()
        self._G.train()
        return output

    def train(self, n_epochs: int, next_letter_to_add: str, use_soft_labels: bool = False):
        current_char_index = character_to_index_mapping[next_letter_to_add]  # 'A' is already present
        max_GPU_memory = 0
        print('Starting epochs, GPU memory in use '
              'before loading the inputs: {} MB'.format(torch.cuda.memory_allocated(torch.cuda.current_device())/1e6))

        # prepare image transform to plot in Tensorboard
        new_image_height = (rectangle_shape[0] - SUP_REMOVE_WIDTH - INF_REMOVE_WIDTH)*IMAGE_WIDTH//rectangle_shape[1]
        produce_transform = Compose([ToPILImage(), Resize((new_image_height, IMAGE_WIDTH)), ToTensor()])

        # prepare fixed points in latent space
        letters_to_watch = list(character_to_index_mapping.keys())
        fixed_latent_points = torch.from_numpy(randn(2 * len(letters_to_watch), NOISE_LENGTH)).to(self._device)
        character_condition = torch.from_numpy(character_to_one_hot(2 * letters_to_watch))
        style_P = torch.zeros((len(letters_to_watch), 1), dtype=torch.double)
        style_G = torch.ones((len(letters_to_watch), 1), dtype=torch.double)
        styles = torch.cat([style_P, style_G], dim=0)
        fixed_conditioning_inputs = torch.cat([character_condition, styles], dim=1).to(self._device)

        # produced JIT models
        bs = self._dataset_loader.batch_size
        G_traced = torch.jit.trace(self._G, (torch.randn(bs, NOISE_LENGTH).to(self._device),
                                             torch.randn(bs, NUM_CHARS + 1).to(self._device)))
        D_traced = torch.jit.trace(self._D, (torch.randn(bs, 1, IMAGE_HEIGHT, IMAGE_WIDTH).to(self._device),
                                             torch.randn(bs, NUM_CHARS + 1).to(self._device)))

        # Epoch iteration
        step = 0
        for epoch in range(1, n_epochs + 1):

            if epoch % add_character_every == 0 and current_char_index < NUM_CHARS:
                self._dataset.add_character_to_training(list(character_to_index_mapping.keys())[current_char_index])
                current_char_index = current_char_index + 1

            self._G.train()
            self._D.train()

            # Iterate over the dataset
            start_time = time.time()
            for batch_count, (X, c, style) in enumerate(self._dataset_loader):
                step += 1

                zero_label = torch.zeros(len(X)).to(device=self._device)
                one_label = torch.ones(len(X)).to(device=self._device)
                if use_soft_labels:
                    zero_label += 0.25 * torch.rand(len(X)).to(self._device)
                    one_label -= 0.25 * torch.rand(len(X)).to(self._device)

                # Arrange data
                X = X.to(device=self._device)
                c = character_to_one_hot(c)
                c = concatenate([c, style.unsqueeze(-1)], axis=1)
                c = torch.from_numpy(c).to(device=self._device)

                # Reset gradient
                self._D_optim.zero_grad()

                # Sample data
                z = torch.from_numpy(randn(len(X), NOISE_LENGTH)).to(device=self._device)

                # Discriminator forward-loss-backward-update
                G_sample = G_traced(z, c)
                D_real = D_traced(X, c)
                D_fake = D_traced(G_sample.detach(), c)

                D_loss_real = self._D_loss(D_real, one_label)
                D_loss_fake = self._D_loss(D_fake, zero_label)
                D_loss = (D_loss_real + D_loss_fake) / 2

                if D_loss > D_loss_threshold:
                    D_loss.backward()
                    self._D_optim.step(None)

                self._writer.add_scalar("Loss/Discriminator", D_loss.item(), step)
                self._writer.add_scalar("Discriminator response/to real images (average)", D_real.mean().item(),
                                        step)
                self._writer.add_scalar("Discriminator response/to fake images (average)", D_fake.mean().item(),
                                        step)

                # Reset gradient
                self._G_optim.zero_grad()

                # Generator forward-loss-backward-update
                z = torch.from_numpy(randn(len(X), NOISE_LENGTH)).to(self._device)

                G_sample = G_traced(z, c)
                D_fake = D_traced(G_sample, c)

                G_loss = self._G_loss(D_fake, one_label)

                if G_loss > G_loss_threshold:
                    G_loss.backward()
                    self._G_optim.step(None)

                self._writer.add_scalar("Loss/Generator", G_loss.item(), step)

                # Store max allocated GPU memory
                max_GPU_memory = max(max_GPU_memory, torch.cuda.max_memory_allocated(torch.cuda.current_device())/1e6)

                last_char_added = next((char for char, index in character_to_index_mapping.items()
                                        if index == current_char_index - 1), None)
                print('Epoch {:4d}, batch {:3d}/{:3d}, D loss {:4f}, G loss {:4f}, '
                      'max GPU memory allocated {:.2f} MB, last char added: {}'.format(epoch, batch_count + 1,
                                                                                       len(self._dataset_loader),
                                                                                       D_loss, G_loss, max_GPU_memory,
                                                                                       last_char_added), end='\r')

            end_time = time.time()
            print('\nEpoch completed in {:.2f} s'.format(end_time - start_time))

            if epoch % save_every == 0:
                torch.save(self._G, "./data/models/G_{}.pth".format(self._current_datetime))
                torch.save(self._D, "./data/models/D_{}.pth".format(self._current_datetime))

            # produce graphical results
            if epoch % produce_every == 0:
                self._G.eval()

                # re-compute fixed point images
                images = G_traced(fixed_latent_points, fixed_conditioning_inputs)
                for image, letter in zip(images[:NUM_CHARS], letters_to_watch):
                    image = produce_transform(image.cpu().detach())
                    self._writer.add_image("Fixed latent points/" + letter + "_P", image, global_step=epoch)
                for image, letter in zip(images[NUM_CHARS:], letters_to_watch):
                    image = produce_transform(image.cpu().detach())
                    self._writer.add_image("Fixed latent points/" + letter + "_G", image, global_step=epoch)

                # generate random character images
                for i in range(num_characters_to_generate):
                    char = choice(list(random_characters_to_generate))
                    style = choice([0, 1])
                    image = self.generate(char, style, do_print=False)
                    image = produce_transform(image.unsqueeze(0))
                    fig = produce_figure(image, "char: %s, style: %s" % (char, style))
                    self._writer.add_figure("Random characters/%d" % i, fig, global_step=epoch)
