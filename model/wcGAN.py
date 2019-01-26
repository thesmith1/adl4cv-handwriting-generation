import time
import argparse
from os import listdir
from os.path import join, isfile
import datetime
from tensorboardX import SummaryWriter
from numpy.random import randn, choice
from numpy import concatenate

import torch
from torch.optim import Optimizer, RMSprop
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage

from componentsGAN import ConditionalGenerator, ConditionalDiscriminator
from condition_encoding import character_to_one_hot
from data_management.character_dataset import CharacterDataset
from utils.global_vars import *
from utils.image_utils import generate, produce_figure
from model.models import ConditionalDCGANGenerator, ConditionalDCGANDiscriminator

clamp_limit = 1e-2
eta_critic = 5
dataset_path = '../data/big/processed/'
labels_file = '../data/big/labels.txt'
models_path = '../data/models/'
logs_path = '../model/runs/'


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

    def train(self, n_epochs: int, next_letter_to_add: str):
        current_char_index = character_to_index_mapping[next_letter_to_add]  # ' ' is already present
        max_GPU_memory = 0
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
            for batch_count, (X, labels, style) in enumerate(self._dataset_loader):

                # batch norm does not allow bs=1, so we check
                if len(X) == 1:
                    continue
                else:
                    step += 1

                # Arrange data
                X = X.to(device=self._device)
                char_conditioning = character_to_one_hot(labels)
                c = concatenate([char_conditioning, style.unsqueeze(-1)], axis=1)
                c = torch.from_numpy(c).to(device=self._device)
                G_loss = None

                # Reset gradient
                self._D_optim.zero_grad()

                # Sample data
                z = torch.from_numpy(randn(len(X), NOISE_LENGTH)).to(device=self._device)

                # Discriminator forward-loss-backward-update
                G_sample = G_traced(z, c)
                D_real = D_traced(X, c)
                D_fake = D_traced(G_sample.detach(), c)
                D_loss = - (torch.mean(D_real) - torch.mean(D_fake))

                if D_loss > D_loss_threshold:
                    D_loss.backward()
                    self._D_optim.step(None)
                    for param in self._D.parameters():
                        param.data.clamp_(-clamp_limit, clamp_limit)

                self._writer.add_scalar("Loss/Discriminator", D_loss.item(), step)
                self._writer.add_scalar("Discriminator response/to real images (average)", D_real.mean().item(),
                                        step)
                self._writer.add_scalar("Discriminator response/to fake images (average)", D_fake.mean().item(),
                                        step)

                if batch_count % eta_critic == 0:
                    # Reset gradient
                    self._G_optim.zero_grad()

                    # Generator forward-loss-backward-update

                    z = torch.from_numpy(randn(len(X), NOISE_LENGTH)).to(self._device)
                    G_sample = G_traced(z, c)
                    D_fake = D_traced(G_sample, c)
                    G_loss = - torch.mean(D_fake)
                    if G_loss > G_loss_threshold:
                        G_loss.backward()
                        self._G_optim.step(None)

                    self._writer.add_scalar("Loss/Generator", G_loss.item(), step)

                # Store max allocated GPU memory
                max_GPU_memory = max(max_GPU_memory, torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1e6)

                last_char_added = next((char for char, index in character_to_index_mapping.items()
                                        if index == current_char_index - 1), None)
                print('Epoch {:4d}, batch {:3d}/{:3d}, D loss {}, G loss {}, '
                      'max GPU memory allocated {:.2f} MB, last char added: {}'.format(epoch, batch_count + 1,
                                                                                       len(self._dataset_loader),
                                                                                       D_loss, G_loss, max_GPU_memory,
                                                                                       last_char_added), end='\r')

            end_time = time.time()
            print('\nEpoch completed in {:.2f} s'.format(end_time - start_time))

            if epoch % save_every == 0:
                print("Saving...", end='')
                torch.save(self._G, "../data/models/W_G_{}.pt".format(str(self._current_datetime)))
                torch.save(self._D, "../data/models/W_D_{}.pt".format(str(self._current_datetime)))
                print("done.")

            # produce graphical results
            if epoch % produce_every == 0:

                print("Producing evaluation results...", end='')
                self._G.eval()

                # re-compute fixed point images
                images = G_traced(fixed_latent_points, fixed_conditioning_inputs)
                for image, letter in zip(images[:NUM_CHARS], letters_to_watch):
                    image = finalizing_transform(image.cpu().detach())
                    self._writer.add_image("Fixed latent points/" + letter + "_P", image, global_step=epoch)
                for image, letter in zip(images[NUM_CHARS:], letters_to_watch):
                    image = finalizing_transform(image.cpu().detach())
                    self._writer.add_image("Fixed latent points/" + letter + "_G", image, global_step=epoch)

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
                        self._writer.add_figure("Random characters/%d" % i, fig, global_step=epoch)

                print("done.")


if __name__ == '__main__':
    # initial set up
    p = argparse.ArgumentParser(prog="python train.py", description="Train GAN for handwriting generation")
    p.add_argument('-m', '--model', help="Allows to start the training from an existing model",
                   type=str, default=None)
    p.add_argument('-a', '--add', help="Allows to specify the last letter added in the dataset on the previous train",
                   type=str, default=None)
    args = p.parse_args()

    current_datetime = str(datetime.datetime.now())
    writer = SummaryWriter(log_dir=join(logs_path, current_datetime))
    next_letter_to_add = 'A'

    # Set device
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')

    # Init models
    if args.model:
        print("Loading models...")
        d_models = [f for f in listdir(models_path) if
                    isfile(join(models_path, f)) and f.endswith('.pt') and f[0] == 'D']
        g_models = [f for f in listdir(models_path) if
                    isfile(join(models_path, f)) and f.endswith('.pt') and f[0] == 'G']
        d_path = ''
        g_path = ''
        for d in d_models:
            if args.model in d:
                d_path = join(models_path, d)
                break
        for g in g_models:
            if args.model in g:
                g_path = join(models_path, g)
                break
        if d_path != '' and g_path != '':
            g = torch.load(g_path)
            d = torch.load(d_path)
            current_datetime = d_path.split('/')[-1][2:-3]  # models MUST be .pt, not .pth
            print('Loaded {} and {}'.format(d_path, g_path))
        else:
            raise Exception('Could not find the models')
    else:
        g = ConditionalDCGANGenerator()
        d = ConditionalDCGANDiscriminator(wasserstein_output=True)

    # Init optimizers
    print('Initializing the optimizers...')
    g_adam = RMSprop(g.parameters(), lr=5e-5, weight_decay=dis_l2_reg)
    d_adam = RMSprop(d.parameters(), lr=5e-5, weight_decay=gen_l2_reg)

    # Load the dataset
    print("Loading dataset...")
    transform = Compose([Resize((IMAGE_HEIGHT, IMAGE_WIDTH)), ToTensor()])
    char_ds = CharacterDataset(dataset_path, labels_file, transform)
    loader = DataLoader(char_ds, batch_size=64, shuffle=True)

    # Restore the content of the dataset if the training is not new
    if args.add:
        next_letter_to_add = args.add
        last_char_added_index = character_to_index_mapping[args.add]
        for curr_char in character_to_index_mapping.keys():
            if curr_char == ' ':  # the space is already present
                pass
            if curr_char == next_letter_to_add:
                break
            else:
                char_ds.add_character_to_training(curr_char)

    # Train
    print("Initiating GAN...")
    gan = WCGAN(g, d, G_optim=g_adam, D_optim=d_adam,
                dataset_loader=loader, dataset=char_ds, device=dev,
                writer=writer, current_datetime=current_datetime)
    gan.train(add_character_every * len(character_to_index_mapping) + 1000,
              next_letter_to_add=next_letter_to_add)
