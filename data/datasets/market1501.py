import os
import os.path as osp
import re
import glob
from . import base


__all__ = [ "Market1501" ]


class Market1501(base.ReidImageDataset):
    """Market1501 dataset

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    Dataset statistics:
        identities: 1501 (+1 for background)
        images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'
    dataset_url = 'https://www.dropbox.com/s/5gypc42z6vdbbtu/Market-1501-v15.09.15.zip?dl=1'

    def __init__(self, root, **kwargs):
        # Download dataset if needed
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, Market1501.dataset_dir)
        self.download_dataset(self.dataset_dir, Market1501.dataset_url)

        # Change dataset directory to the extracted directory
        self.dataset_dir = osp.join(self.dataset_dir, 'Market-1501-v15.09.15')

        # Essential directories
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        # Process essential directories
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        super().__init__(train, query, gallery, **kwargs)

    def _process_dir(self, target_dir, relabel=False):
        img_paths = glob.glob(osp.join(target_dir, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        # Build pid relabeling table
        pid_set = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            pid_set.add(pid)
        pid_to_label = { pid:label for label, pid in enumerate(pid_set) }

        # Build list of data
        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            assert 0 <= pid <= 1501
            assert 1 <= camid <= 6
            camid -= 1
            if relabel:
                pid = pid_to_label[pid]
            data.append((img_path, pid, camid))

        return data

class Market1501Pose(base.ReidPoseDataset):
    """Market1501 dataset with additional pose image

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    Dataset statistics:
        identities: 1501 (+1 for background)
        images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'
    dataset_url = 'https://www.dropbox.com/s/5gypc42z6vdbbtu/Market-1501-v15.09.15.zip?dl=1'

    def __init__(self, root, **kwargs):
        # Download dataset if needed
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, Market1501.dataset_dir)
        self.download_dataset(self.dataset_dir, Market1501.dataset_url)

        # Change dataset directory to the extracted directory
        self.dataset_dir = osp.join(self.dataset_dir, 'Market-1501-v15.09.15')

        # Essential directories
        self.train_img_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.train_pose_dir = osp.join(self.dataset_dir, 'bounding_box_train_pose')

        self.query_img_dir = osp.join(self.dataset_dir, 'query')
        self.query_pose_dir = osp.join(self.dataset_dir, 'query_pose')

        self.gallery_img_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.gallery_pose_dir = osp.join(self.dataset_dir, 'bounding_box_test_pose')

        # Process essential directories
        train = self._process_dir(self.train_img_dir,
                                self.train_pose_dir,
                                relabel=True)
        query = self._process_dir(self.query_img_dir,
                                self.query_pose_dir,
                                relabel=False)
        gallery = self._process_dir(self.gallery_img_dir,
                                self.gallery_pose_dir,
                                relabel=False)

        super().__init__(train, query, gallery, **kwargs)

    def _process_dir(self, img_dir, pose_dir, relabel=False):
        # Consider only files both in img_dir and pose_dir
        img_paths = [ f for f in os.listdir(img_dir) if 'jpg' in f ]
        pose_paths = [ f for f in os.listdir(pose_dir) if 'jpg' in f ]
        common_paths = sorted(list(set(img_paths).intersection(set(pose_paths))))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        # Build pid relabeling table
        pid_set = set()
        for path in common_paths:
            pid, _ = map(int, pattern.search(path).groups())
            if pid == -1:
                continue
            pid_set.add(pid)
        pid_to_label = { pid:label for label, pid in enumerate(pid_set) }

        # Build list of data
        data = []
        for path in common_paths:
            pid, camid = map(int, pattern.search(path).groups())
            if pid == -1:
                continue
            assert 0 <= pid <= 1501
            assert 1 <= camid <= 6
            camid -= 1
            if relabel:
                pid = pid_to_label[pid]

            img_path = osp.join(img_dir, path)
            pose_path = osp.join(pose_dir, path)
            data.append((img_path, pose_path, pid, camid))

        return data
