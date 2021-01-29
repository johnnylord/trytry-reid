import os
import os.path as osp
import zipfile
import tarfile
import copy
from PIL import Image
from .. import utils

__all__ = [ "ReidDataset", "ReidImageDataset" ]


class ReidDataset:
    """Base class for all reid dataset

    ReidDataset can be divided into three categories:
        1. train
        2. query
        3. gallery

    `query` and `gallery` dataset are from the same pid space. `train` has its
    own pid space. ReidDataset will organize these three dataset, and pack them
    into a single dataset.

    Attributes:
        train (list of tuple):  reid dataset from train source
        query (list of tuple):  reid dataset from query source
        gallery (list of tuple):reid dataset from gallery source
        transform (function):   tranformation function
        mode (str): data source
    """
    def __init__(self, train, query, gallery,
                transform=None,
                mode='train'):
        self.train = train
        self.query = query
        self.gallery = gallery
        self.transform = transform
        self.mode = mode

        # Record pids and cams information
        self.num_train_pids = self._get_num_pids(train)
        self.num_train_cams = self._get_num_cams(train)

        # Aggregate different sources of data
        if mode == 'train':
            self.data = train
        elif mode == 'query':
            self.data = query
        elif mode == 'gallery':
            self.data = gallery
        elif mode == 'all':
            self.data = self._combine_all()
        else:
            raise ValueError("'mode' cannot be {}".format(mode))

    def __add__(self, other):
        """Combine training datasets together"""
        train = copy.deepcopy(self.train)
        for img_path, pid, camid in other.train:
            pid += self.num_train_pids
            camid += self.num_train_cams
            train.append((img_path, pid, camid))

        return ReidImageDataset(train,
                                self.query,
                                self.gallery,
                                transform=self.transform,
                                mode='train')

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        summary = "[{}]\n" \
                  "| Source | Images | Pids | Camids |\n" \
                  "===================================\n" \
                  "| train  | {:^6} | {:^4} | {:^6} |\n" \
                  "| query  | {:^6} | {:^4} | {:^6} |\n" \
                  "| gallery| {:^6} | {:^4} | {:^6} |\n".format(self.__class__.__name__,
                    len(self.train), self._get_num_pids(self.train), self._get_num_cams(self.train),
                    len(self.query), self._get_num_pids(self.query), self._get_num_cams(self.query),
                    len(self.gallery), self._get_num_pids(self.gallery), self._get_num_cams(self.gallery))
        return summary

    def __len__(self):
        return len(self.data)

    def download_dataset(self, dataset_dir, dataset_url):
        """Download dataset to the specified directory

        Args:
            dataset_dir (str): dataset directory
            dataset_url (str): url to download dataset
        """
        if osp.exists(dataset_dir):
            return

        if dataset_url is None:
            raise RuntimeError(
                    '{} dataset needs to be manually prepared, '
                    'please follow the document to prepare '
                    'this dataset'.format(self.__class__.__name__))

        os.makedirs(dataset_dir)
        fpath = osp.join(dataset_dir, osp.basename(dataset_url))
        if 'dropbox' in dataset_url:
            fpath = fpath.split("?")[0]
            utils.download_from_dropbox(dataset_url, fpath)
        else:
            utils.download_from_url(dataset_url, fpath)

        if fpath.endswith('tar'):
            tar = tarfile.open(fpath)
            tar.extractall(path=dataset_dir)
            tar.close()
        elif fpath.endswith('zip'):
            zip_ = zipfile.ZipFile(fpath, 'r')
            zip_.extractall(dataset_dir)
            zip_.close()
        else:
            raise RuntimeError("Don't know how to extract '{}'".format(fpath))

        os.remove(fpath)

    def _get_num_pids(self, data):
        return len(set([ pid for _, pid, _ in data ]))

    def _get_num_cams(self, data):
        return len(set([ camid for _, _, camid in data ]))

    def _combine_all(self):
        """Combines train, query, gallery together for training"""
        combined = copy.deepcopy(self.train)

        # relabel pids in gallery (query shares the same scope)
        gallery_pid_set = set()
        for _, pid, _ in self.gallery:
            gallery_pid_set.add(pid)

        pid_to_label = { pid: label for label, pid in enumerate(gallery_pid_set) }

        def combine_data(data):
            for img_path, pid, camid in data:
                pid = pid_to_label[pid] + self.num_train_pids
                combined.append((img_path, pid, camid))

        combine_data(self.query)
        combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self._get_num_pids(self.train)
        self.num_train_cams = self._get_num_cams(self.train)

        return self.train


class ReidImageDataset(ReidDataset):
    """Base class for reid image dataset """

    def __init__(self, train, query, gallery, **kwargs):
        super().__init__(train, query, gallery, **kwargs)

    def __getitem__(self, idx):
        img_path, pid, camid = self.data[idx]

        # Read in PIL image
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path
