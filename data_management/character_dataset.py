import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
from PIL.Image import open


class CharacterDataset(Dataset):
    def __init__(self, img_folder_path: str, labels_file_path: str, transform: Compose):
        self._images = []
        self._labels = []
        self._styles = []
        self._images_names = []
        self._img_folder_path = img_folder_path
        self._labels_file_path = labels_file_path
        self._transform = transform
        self.load_labels()
        assert len(self._images) == len(self._labels) == len(self._styles)

    def __getitem__(self, index):
        return self._images[index], self._labels[index], self._styles[index]

    def __len__(self):
        return len(self._labels)

    def load_labels(self):
        # Load the labels file
        file_content = np.loadtxt(self._labels_file_path, delimiter=' ', dtype=str)
        self._images_names = file_content[:, 0]
        self._labels = file_content[:, 1]
        self._styles = file_content[:, 2].astype(int)
        # Load the images
        for img_path in self._images_names:
            try:
                img = open(self._img_folder_path + img_path)
                img = self._transform(img)
                self._images.append(img)
            except FileNotFoundError as e:
                print(e)
        # Substitute the labels with _ with spaces
        for idx, lab in enumerate(self._labels):
            if lab == '_':
                self._labels[idx] = ' '


if __name__ == '__main__':
    d = CharacterDataset('../data/processed/', '../data/labels_test.txt', Compose([ToTensor()]))
    loader = DataLoader(d, batch_size=3, shuffle=True)
    for epoch in range(100):
        for batch in loader:
            pass
