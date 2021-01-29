import numpy as np
from torch.utils.data.sampler import Sampler


class BalancedBatchSampler(Sampler):

    def __init__(self, labels, P, K):
        self.labels = labels
        self.n_samples = len(labels)
        self.P = P
        self.K = K
        self.batch_size = P*K

        # Construct lookup table
        self.label_set = list(set(labels))
        self.label_to_indices = { label: np.where(np.array(labels) == label)[0]
                                for label in self.label_set }
        for l in self.label_set:
            np.random.shuffle(self.label_to_indices[l])

        # dynamic information
        self.used_label_indices_count = { label: 0 for label in self.label_set }
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count + self.batch_size > self.n_samples:
            raise StopIteration

        target_labels = np.random.choice(self.label_set, self.P, replace=False)
        indices = []
        for target_label in target_labels:
            search_ptr = self.used_label_indices_count[target_label]
            indices.extend(self.label_to_indices[target_label][search_ptr:search_ptr+self.K])
            self.used_label_indices_count[target_label] += self.K

            if self.used_label_indices_count[target_label] + self.K > len(self.label_to_indices[target_label]):
                np.random.shuffle(self.label_to_indices[target_label])
                self.used_label_indices_count[target_label] = 0

            self.count += self.batch_size

        return indices

    def __len__(self):
        return self.n_samples // self.batch_size
