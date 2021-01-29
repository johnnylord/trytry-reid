from torch.utils.data import random_split, ConcatDataset
from torchvision.datasets import MNIST as MNIST_

class MNIST:

    def __init__(self, root, transform=None, mode='train'):
        self.root = root
        self.transform = transform
        self.mode = mode

        # Load dataset
        train_data = MNIST_(root=root, train=True)
        train_samples = int(len(train_data)*0.7)
        valid_samples = int(len(train_data)*0.3)

        # Split datasets to three different domains
        train, query = random_split(train_data, [train_samples, valid_samples])
        gallery = MNIST_(root=root, train=False)

        if mode == 'train':
            self.data = train
        elif mode == 'query':
            self.data = query
        elif mode == 'gallery':
            self.data = gallery
        elif mode == 'all':
            self.data = ConcatDataset([train, query, gallery])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, 0, ""
