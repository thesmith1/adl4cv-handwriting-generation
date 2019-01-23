import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision.models import inception_v3
import argparse
import os
from cv2 import resize, imread
from utils.global_vars import *
from utils.condition_encoding import character_to_one_hot
import numpy as np
from scipy.linalg import sqrtm

DEF_DATASET_FOLDER = './data/big/processed'
DEF_LABEL_FILE = './data/big/labels.txt'
iterations = 100
batch_size = 24


def compute_features(self, x):
    if self.transform_input:
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # 299 x 299 x 3
    x = self.Conv2d_1a_3x3(x)
    # 149 x 149 x 32
    x = self.Conv2d_2a_3x3(x)
    # 147 x 147 x 32
    x = self.Conv2d_2b_3x3(x)
    # 147 x 147 x 64
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 73 x 73 x 64
    x = self.Conv2d_3b_1x1(x)
    # 73 x 73 x 80
    x = self.Conv2d_4a_3x3(x)
    # 71 x 71 x 192
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 35 x 35 x 192
    x = self.Mixed_5b(x)
    # 35 x 35 x 256
    x = self.Mixed_5c(x)
    # 35 x 35 x 288
    x = self.Mixed_5d(x)
    # 35 x 35 x 288
    x = self.Mixed_6a(x)
    # 17 x 17 x 768
    x = self.Mixed_6b(x)
    # 17 x 17 x 768
    x = self.Mixed_6c(x)
    # 17 x 17 x 768
    x = self.Mixed_6d(x)
    # 17 x 17 x 768
    x = self.Mixed_6e(x)
    # 17 x 17 x 768
    x = self.Mixed_7a(x)
    # 8 x 8 x 1280
    x = self.Mixed_7b(x)
    # 8 x 8 x 2048
    x = self.Mixed_7c(x)
    # 8 x 8 x 2048
    x = F.avg_pool2d(x, kernel_size=8)
    # 1 x 1 x 2048
    x = F.dropout(x, training=self.training)
    # 1 x 1 x 2048
    x = x.view(x.size(0), -1)
    return x


class DatasetFromFolder(Dataset):

    def __init__(self, root, transform):
        self._root = root
        self._transform = transform
        self._images = [imread(os.path.join(root, image_filename)) for image_filename in os.listdir(root)]

    def __getitem__(self, idx):
        image = self._images[idx]
        return self._transform(image)

    def __len__(self):
        return len(self._images)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str, help='allows to specify the model to evaluate')
    parser.add_argument('--dataset-folder', '-d', type=str, default=DEF_DATASET_FOLDER,
                        help='allows to specify the location of the original data')
    parser.add_argument('--label-file', '-l', type=str, default=DEF_LABEL_FILE,
                        help='allows to specify the label file of the dataset, '
                             'to generate samples that match the data distribution')
    parser.add_argument('--precomputed-features', '-p', type=str, default=None,
                        help='allows to specify an input .npy file to load pre-computed features')
    args = parser.parse_args()
    return args


def main(args):
    print("Loading model to evaluate...", end='')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(args.model_file)
    model.to(device=device)
    model.eval()
    print('done.')

    print('Loading Inception model...', end='')
    inception_model = inception_v3(pretrained=True)
    inception_model.to(device)
    inception_model.eval()
    print('done.')

    print('Preparing data loaders...', end='')
    image_transform = Compose([lambda x: resize(x, (299, 299)), ToTensor()])
    real_image_dataset = DatasetFromFolder(root=arguments.dataset_folder, transform=image_transform)
    real_image_loader = DataLoader(real_image_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print('done.')

    print('Processing label file...', end='')
    label_file = np.loadtxt(args.label_file, delimiter=' ', dtype=str)[:, 1:4]
    label_file[label_file == '_'] = ' '
    character_occurrences = dict()
    for label_file_row in label_file:
        key = ''.join(label_file_row)
        character_occurrences[key] = character_occurrences.get(key, 0) + 1
    for key in character_occurrences:
        character_occurrences[key] /= len(label_file)
    possible_character_combinations = list(character_occurrences.keys())
    character_occurrences = list(character_occurrences.values())
    print('done.')

    if args.precomputed_features is None:
        print("Producing Inception features for real data...", end='')
        real_features = []
        for i, batch in enumerate(real_image_loader):
            features_batch = compute_features(inception_model, batch.to(device))
            real_features.append(features_batch.cpu().detach().numpy())
        real_features = np.concatenate(real_features, axis=0)
        # np.save('./data/big/inception_features.npy', real_features)
    else:
        print('Loading Inception features for real data from specified file...', end='')
        real_features = np.load(args.precomputed_features)
    mu_real = np.mean(real_features, axis=0)
    cov_real = np.cov(real_features, rowvar=False)
    print('done.')

    print('Starting evaluation...')
    generated_features = None
    for i in range(iterations):

        # produce generated images
        random_conditioning_characters = np.random.choice(possible_character_combinations,
                                                          p=character_occurrences, size=batch_size)
        random_conditioning_characters = np.concatenate([character_to_one_hot(list(random_conditioning_group))
                                                         for random_conditioning_group in
                                                         random_conditioning_characters], axis=0)
        random_conditioning_characters = torch.from_numpy(random_conditioning_characters)
        random_styles = torch.from_numpy(np.random.choice([0., 1.], size=(batch_size, 1)))

        conditioning_input = torch.cat([random_conditioning_characters, random_styles], dim=1).to(device)
        z = torch.randn(batch_size, NOISE_LENGTH)
        generated_batch = model(z, conditioning_input)
        generated_batch = generated_batch.cpu().detach().numpy()

        # produce generated features
        generated_batch = np.transpose(generated_batch, [0, 2, 3, 1])
        generated_batch = np.repeat(generated_batch, repeats=3, axis=3)
        generated_images = [image_transform(generated_image) for generated_image in generated_batch]
        generated_images = torch.from_numpy(np.stack(generated_images)).to(device)
        generated_features_batch = compute_features(inception_model, generated_images).cpu().detach().numpy()

        if generated_features is not None:
            generated_features = np.concatenate([generated_features, generated_features_batch], axis=0)
        else:
            generated_features = generated_features_batch

        # compute FID score
        mu_generated = np.mean(generated_features, axis=0)
        cov_generated = np.cov(generated_features, rowvar=False)
        fid_score = np.linalg.norm(mu_real - mu_generated)**2 + \
                    np.trace(cov_real + cov_generated - 2*sqrtm(cov_real.dot(cov_generated), disp=False)[0])
        print("Iteration %d, FID: %f, generated images: %d" % (i, fid_score.real, len(generated_features)))


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
