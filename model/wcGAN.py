import time
import sys
import os
import datetime
from tensorboardX import SummaryWriter
from numpy.random import randn, choice
from numpy import concatenate

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage

from componentsGAN import ConditionalGenerator, ConditionalDiscriminator


lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)
ext_lib_path = os.path.abspath(os.path.join(__file__, '../../utils'))
sys.path.append(ext_lib_path)

from data_management.character_dataset import CharacterDataset
from utils.condition_encoding import character_to_one_hot
from utils.global_vars import *
from utils.image_utils import generate, produce_figure


clamp_limit = 1e-2
eta_critic = 5


class WCGAN:
    def __init__(self, G: ConditionalGenerator, D: ConditionalDiscriminator,
                 G_optim: Optimizer, D_optim: Optimizer,
                 dataset_loader: DataLoader,
                 dataset: CharacterDataset,
                 writer: SummaryWriter,
                 device: torch.device,
                 current_datetime: str = None):
        self._G = G
        self._D = D
        self._G_optim = G_optim
        self._D_optim = D_optim
        self._device = device
        self._dataset_loader = dataset_loader
        self._dataset = dataset
        self._writer = writer
        self._G.to(device=self._device)
        self._D.to(device=self._device)
        if current_datetime:
            self._current_datetime = current_datetime
        else:
            self._current_datetime = datetime.datetime.now()

    def train(self, n_iterations: int, next_letter_to_add: str):
        current_char_index = character_to_index_mapping[next_letter_to_add]  # ' ' is already present
        print('Starting epochs, GPU memory in use '
              'before loading the inputs: {} MB'.format(torch.cuda.memory_allocated(torch.cuda.current_device()) / 1e6))

        # prepare image transform to plot in TensorBoard
        final_image_height = (rectangle_shape[0] - SUP_REMOVE_WIDTH - INF_REMOVE_WIDTH) * IMAGE_WIDTH // \
                             rectangle_shape[1]
        finalizing_transform = Compose([ToPILImage(), Resize((final_image_height, IMAGE_WIDTH)), ToTensor()])

        # prepare fixed points in latent space
        letters_to_watch = list(character_to_index_mapping.keys())
        fixed_latent_points = torch.from_numpy(randn(2 * len(letters_to_watch), NOISE_LENGTH)).to(self._device)
        zero_conditioning = tuple([' ' for _ in range(2 * len(letters_to_watch))])
        current_char_conditioning = tuple(2 * letters_to_watch)
        character_condition = [zero_conditioning, current_char_conditioning, zero_conditioning]
        character_condition = torch.from_numpy(character_to_one_hot(character_condition))
        style_P = torch.zeros((len(letters_to_watch), 1), dtype=torch.double)
        style_G = torch.ones((len(letters_to_watch), 1), dtype=torch.double)
        styles = torch.cat([style_P, style_G], dim=0)
        fixed_conditioning_inputs = torch.cat([character_condition, styles], dim=1).to(self._device)

        # produce JIT models
        bs = self._dataset_loader.batch_size
        G_traced = torch.jit.trace(self._G, (torch.randn(bs, NOISE_LENGTH).to(self._device),
                                             torch.randn(bs, NUM_CHARS * 3 + 1).to(self._device)))
        D_traced = torch.jit.trace(self._D, (torch.randn(bs, 1, IMAGE_HEIGHT, IMAGE_WIDTH).to(self._device),
                                             torch.randn(bs, NUM_CHARS * 3 + 1).to(self._device)))

        # Train iteration
        for iteration in range(1, n_iterations + 1):

            labels, style = None, None
            if iteration % add_character_every == 0 and current_char_index < NUM_CHARS:
                self._dataset.add_character_to_training(list(character_to_index_mapping.keys())[current_char_index])
                current_char_index = current_char_index + 1

            start_time = time.time()

            self._G.train()
            self._D.train()

            # ---------------------TRAIN D------------------------
            D_loss = D_real = D_fake = None
            D_losses = []
            for i in range(1, eta_critic + 1):

                # sample batch
                X, labels, style = next(iter(self._dataset_loader))

                # batch norm does not allow bs=1, so we check
                if len(X) == 1:
                    continue

                # Arrange data
                X = X.to(device=self._device)
                char_conditioning = character_to_one_hot(labels)
                c = concatenate([char_conditioning, style.unsqueeze(-1)], axis=1)
                c = torch.from_numpy(c).to(device=self._device)

                # Reset gradient
                self._D_optim.zero_grad()

                # Sample z
                z = torch.from_numpy(randn(len(X), NOISE_LENGTH)).to(device=self._device)

                # Discriminator forward-loss-backward-update
                G_sample = G_traced(z, c)
                D_real = D_traced(X, c)
                D_fake = D_traced(G_sample.detach(), c)
                D_loss = - (torch.mean(D_real) - torch.mean(D_fake))
                D_losses.append(D_loss.item())

                D_loss.backward()
                self._D_optim.step(None)
                for param in self._D.parameters():
                    param.data.clamp_(-clamp_limit, clamp_limit)

            print('D Iteration {:5d}, D losses {}'.format(iteration, D_losses))

            self._writer.add_scalar("Loss/Discriminator", D_loss.item(), iteration)
            self._writer.add_scalar("Critic response/to real images (average)", D_real.mean().item(), iteration)
            self._writer.add_scalar("Critic response/to fake images (average)", D_fake.mean().item(), iteration)

            # ---------------------TRAIN G------------------------
            self._G_optim.zero_grad()

            # prepare G inputs
            char_conditioning = character_to_one_hot(labels)
            c = concatenate([char_conditioning, style.unsqueeze(-1)], axis=1)
            c = torch.from_numpy(c).to(device=self._device)

            # Generator forward-loss-backward-update
            z = torch.from_numpy(randn(len(char_conditioning), NOISE_LENGTH)).to(self._device)
            G_sample = G_traced(z, c)
            D_fake = D_traced(G_sample, c)
            G_loss = - torch.mean(D_fake)
            G_loss.backward()
            self._G_optim.step(None)

            self._writer.add_scalar("Loss/Generator", G_loss.item(), iteration)

            last_char_added = next((char for char, index in character_to_index_mapping.items()
                                    if index == current_char_index - 1), None)
            end_time = time.time()
            print('G Iteration {:5d}, G loss {:4f}, time/iteration: {:3f}s, '
                  'last char added: {}'.format(iteration, G_loss, end_time - start_time, last_char_added))
            # save
            if iteration % save_every == 0:
                print("Saving...", end='')
                torch.save(self._G, "./data/models/G_W_{}.pt".format(str(self._current_datetime)))
                torch.save(self._D, "./data/models/D_W_{}.pt".format(str(self._current_datetime)))
                print("done.")

            # produce graphical results
            if iteration % produce_every == 0:

                print("Producing evaluation results...", end='')
                self._G.eval()

                # re-compute fixed point images
                images = G_traced(fixed_latent_points, fixed_conditioning_inputs)
                for image, letter in zip(images[:NUM_CHARS], letters_to_watch):
                    image = finalizing_transform(image.cpu().detach())
                    self._writer.add_image("Fixed latent points/" + letter + "_P", image, global_step=iteration)
                for image, letter in zip(images[NUM_CHARS:], letters_to_watch):
                    image = finalizing_transform(image.cpu().detach())
                    self._writer.add_image("Fixed latent points/" + letter + "_G", image, global_step=iteration)

                # generate random character images
                random_characters_to_generate = list(key for key, value in character_to_index_mapping.items()
                                                     if value < current_char_index - 1) + [' ']
                if random_characters_to_generate:
                    for i in range(num_characters_to_generate):
                        character_conditioning = (' ', choice(list(random_characters_to_generate)), ' ')
                        style = choice([0, 1])
                        image = generate(self._G, character_conditioning, style, device=self._device)
                        image = finalizing_transform(image.unsqueeze(0))
                        fig = produce_figure(image, "prev: {}, curr: {}, "
                                                    "next: {}, style: {}".format(*character_conditioning, style))
                        self._writer.add_figure("Random characters/%d" % i, fig, global_step=iteration)

                print("done.")
