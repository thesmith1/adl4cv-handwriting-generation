import argparse
import os
import sys
from os import listdir
from os.path import isfile, join
from datetime import datetime

import torch
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize

lib_path = os.path.abspath(os.path.join(__file__, '.'))
sys.path.append(lib_path)

from data_management.character_dataset import CharacterDataset
from global_vars import *
from model.models import ConditionalDCGANDiscriminator, ConditionalDCGANGenerator
from model.cGAN import CGAN

dataset_path = './data/big/processed/'
labels_file = './data/big/labels.txt'
models_path = './data/models/'
logs_path = './model/runs/'

if __name__ == '__main__':
    p = argparse.ArgumentParser(prog="python train.py", description="Train GAN for handwriting generation")
    p.add_argument('-m', '--model', help="Allows to start the training from an existing model",
                   type=str, default=None)
    args = p.parse_args()

    current_datetime = str(datetime.now())

    # Set device
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')

    # Init models
    if args.model:
        d_models = [f for f in listdir(models_path) if
                    isfile(join(models_path, f)) and f.endswith('.pth') and f[0] == 'D']
        g_models = [f for f in listdir(models_path) if
                    isfile(join(models_path, f)) and f.endswith('.pth') and f[0] == 'G']
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
            current_datetime = d_path.split('/')[-1][2:-4]  # models MUST be .pth, not .pt
            print('Loaded {} and {}'.format(d_path, g_path))
        else:
            raise Exception('Could not find the models')
    else:
        g = ConditionalDCGANGenerator()
        d = ConditionalDCGANDiscriminator()

    # Init optimizers
    g_adam = Adam(g.parameters(), lr=1e-4)
    d_adam = Adam(d.parameters(), lr=1e-4)

    # Load the dataset
    transform = Compose([Resize((IMAGE_HEIGHT, IMAGE_WIDTH)), ToTensor()])
    char_ds = CharacterDataset(dataset_path, labels_file, transform)
    loader = DataLoader(char_ds, batch_size=32, shuffle=True)

    # Train
    writer = SummaryWriter(log_dir=join(logs_path, current_datetime))
    gan = CGAN(g, d, BCELoss(), BCELoss(), G_optim=g_adam, D_optim=d_adam,
               dataset_loader=loader, dataset=char_ds, device=dev,
               writer=writer, current_datetime=current_datetime)
    gan.train(20000)

