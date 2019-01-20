import numpy as np
from random import choice
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from PIL.Image import open


class CharacterDataset(Dataset):
    def __init__(self, img_folder_path: str, labels_file_path: str, transform: Compose):
        self._images = []
        self._labels = []
        self._styles = []
        self._images_names = []
        self._training_images = []  # images that are already used during training (one character at a time training)
        self._img_folder_path = img_folder_path
        self._labels_file_path = labels_file_path
        self._transform = transform
        self.load_labels()
        self.add_character_to_training(' ')
        assert len(self._images) == len(self._labels) == len(self._styles)

    def __getitem__(self, index):
        img, lab, stl = self._training_images[index]
        return img, lab, stl

    def __len__(self):
        return len(self._training_images)

    def load_labels(self):
        # Load the labels file
        file_content = np.loadtxt(self._labels_file_path, delimiter=' ', dtype=str)
        self._images_names = file_content[:, 0]
        self._labels = file_content[:, 1:4].tolist()
        self._styles = file_content[:, 4].astype(int)
        # Load the images
        for img_path in self._images_names:
            try:
                img = open(self._img_folder_path + img_path)
                img = self._transform(img)
                self._images.append(img)
            except FileNotFoundError as e:
                print(e)
        # Substitute the labels with _ with spaces
        for i, annotations in enumerate(self._labels):
            for j, a in enumerate(annotations):
                if a == '_':
                    self._labels[i][j] = ' '

    def add_character_to_training(self, char: str):
        new_images = []
        for idx, lab in enumerate(self._labels):
            if lab[1] == char:
                new_images.append((self._images[idx], lab, self._styles[idx]))
        self._training_images.extend(new_images)

    def get_sample(self, prev_char: str, curr_char: str, next_char: str, st: int):
        assert(len(prev_char) == len(curr_char) == len(next_char) == 1)
        samples = [s for idx, s in enumerate(self._images)
                   if self._labels[idx] == [prev_char, curr_char, next_char] and self._styles[idx] == st]
        if len(samples) == 0:
            raise ValueError('No samples matches the given specification: {}{}{}'.format(prev_char, curr_char, next_char))
        return choice(samples)


if __name__ == '__main__':
    d = CharacterDataset('../data/big/processed/', '../data/big/labels.txt', Compose([Resize((64, 64)), ToTensor()]))
    loader = DataLoader(d, batch_size=3, shuffle=True)
    for epoch in range(100):
        for batch in loader:
            pass
