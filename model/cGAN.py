import torch
from matplotlib.pyplot import imshow, show, figure
from numpy.random import randn
from numpy import concatenate
from torch.nn.modules.loss import _Loss, BCELoss
from torch.optim import Optimizer, Adam, SGD
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from tensorboardX import SummaryWriter
from componentsGAN import ConditionalGenerator, ConditionalDiscriminator
from condition_encoding import character_to_one_hot
from data_management.character_dataset import CharacterDataset
from global_vars import NOISE_LENGTH, NUM_CHARS, character_to_index_mapping
from models import ConditionalDCGANGenerator, ConditionalDCGANDiscriminator
import time


class CGAN:
    def __init__(self, G: ConditionalGenerator, D: ConditionalDiscriminator,
                 G_loss: _Loss, D_loss: _Loss,
                 G_optim: Optimizer, D_optim: Optimizer,
                 dataset_loader: DataLoader,
                 dataset: CharacterDataset,
                 device: torch.device):
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
        current_character_index = 27  # 0 is already present
        max_GPU_memory = 0
        print('Starting epochs, GPU memory in use '
              'before loading the inputs: {} MB'.format(torch.cuda.memory_allocated(torch.cuda.current_device())/1e6))

        # prepare fixed points in latent space
        letters_to_watch = "Paul"
        fixed_latent_points = torch.from_numpy(randn(len(letters_to_watch), NOISE_LENGTH)).to(device)
        fixed_conditioning_inputs = concatenate([(character_to_one_hot(letter)) for letter in letters_to_watch])
        fixed_conditioning_inputs = torch.from_numpy(fixed_conditioning_inputs).to(device)

        # produced JIT models
        G_traced = torch.jit.trace(self._G, (torch.randn(32, NOISE_LENGTH).to(device),
                                             torch.randn(32, NUM_CHARS).to(device)))
        D_traced = torch.jit.trace(self._D, (torch.randn(32, 1, 64, 64).to(device),
                                             torch.randn(32, NUM_CHARS).to(device)))

        # Epoch iteration
        for epoch in range(1, n_epochs + 1):
            if epoch % 200 == 0 and current_character_index < NUM_CHARS:
                self._dataset.add_character_to_training(list(character_to_index_mapping.keys())[current_character_index])
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
                # G_sample = self._G(z, c)
                G_sample = G_traced(z, c)
                # D_real = self._D(X, c)
                D_real = D_traced(X, c)
                # D_fake = self._D(G_sample.detach(), c)
                D_fake = D_traced(G_sample.detach(), c)

                zero_label = torch.zeros(len(X)).to(device=self._device)
                one_label = torch.ones(len(X)).to(device=self._device)

                D_loss_real = self._D_loss(D_real, one_label)
                D_loss_fake = self._D_loss(D_fake, zero_label)
                D_loss = (D_loss_real + D_loss_fake) / 2

                D_loss.backward()
                self._D_optim.step(None)

                # Reset gradient
                self._G_optim.zero_grad()

                # Generator forward-loss-backward-update
                z = torch.from_numpy(randn(len(X), NOISE_LENGTH)).to(self._device)

                # G_sample = self._G(z, c)
                G_sample = G_traced(z, c)
                # D_fake = self._D(G_sample, c)
                D_fake = D_traced(G_sample, c)

                G_loss = self._G_loss(D_fake, one_label)

                G_loss.backward()
                self._G_optim.step(None)

                # Store max allocated GPU memory
                max_GPU_memory = max(max_GPU_memory, torch.cuda.max_memory_allocated(torch.cuda.current_device())/1e6)

                # Logging batch-wise
                step = (epoch - 1)*len(self._dataset_loader) + batch_count
                writer.add_scalar("Loss/Generator", G_loss.item(), step)
                writer.add_scalar("Loss/Discriminator", D_loss.item(), step)
                writer.add_scalar("Discriminator response/to real images (average)", D_real.mean().item(), step)
                writer.add_scalar("Discriminator response/to fake images (average)", D_fake.mean().item(), step)
                print('Epoch {:2d}, batch {:2d}/{:2d}, D loss {:4f}, G loss {:4f}, '
                      'max GPU memory allocated {:.2f} MB'.format(epoch, batch_count + 1, len(self._dataset_loader),
                                                                  D_loss, G_loss, max_GPU_memory),
                      end='\r')

            end_time = time.time()
            print('\nEpoch completed in {:.2f} s'.format(end_time - start_time))

            if epoch % 100 == 0:
                torch.save(self._G, "../data/models/gen.pth")
                torch.save(self._D, "../data/models/dis.pth")

            # re-compute fixed point images
            self._G.eval()
            images = G_traced(fixed_latent_points, fixed_conditioning_inputs)
            for image, letter in zip(images, letters_to_watch):
                writer.add_image("Fixed latent points/" + letter, image, global_step=epoch)

            if epoch % 10 == 0:
                torch.save(g, "../data/models/gen.pth")
                torch.save(d, "../data/models/dis.pth")


if __name__ == '__main__':
    # Test cGAN class

    # set training device
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # set models
    g = ConditionalDCGANGenerator()
    d = ConditionalDCGANDiscriminator()

    # set optimizers
    g_adam = Adam(g.parameters(), lr=1e-4)
    d_adam = SGD(d.parameters(), lr=1e-3, momentum=0.01)  # momentum and nesterov update seem inefficient

    # load dataset
    transform = Compose([Resize((64, 64)), ToTensor()])
    dataset = CharacterDataset('../data/big/processed/', '../data/big/labels.txt', transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # train
    writer = SummaryWriter()
    gan = CGAN(g, d, BCELoss(), BCELoss(), G_optim=g_adam, D_optim=d_adam, dataset_loader=loader, dataset=dataset,
               device=device)
    gan.train(10000)
