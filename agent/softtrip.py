import os
import os.path as osp
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from data.transform import RandomErasing
from data.datasets import get_dataset_cls
from data.sampler import BalancedBatchSampler
from loss.utils import RandomNegativeTripletSelector
from loss.triplet import OnlineTripletLoss

from model import get_model_cls
from utils.metric import compute_AP_CMC


__all__ = [ "SoftTripAgent" ]

class SoftTripAgent:
    """Train ReID model with CrossEntropy Loss and Triplet loss"""
    def __init__(self, config):

        # Torch environment
        # ======================================================
        self.config = config
        self.device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")

        # Define model
        # =======================================================
        model_cls = get_model_cls(config['model']['name'])
        self.model = model_cls(**config['model']['kwargs'])
        self.model = self.model.to(self.device)

        # Define dataset
        # =======================================================
        tr_transform = transforms.Compose([
            transforms.Resize(config['dataset']['size']),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(config['dataset']['size']),
            transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.3)
            ])
        te_transform = transforms.Compose([
            transforms.Resize(config['dataset']['size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])

        dataset_cls = get_dataset_cls(config['dataset']['name'])
        train_dataset = dataset_cls(config['dataset']['root'],
                                    transform=tr_transform,
                                    mode='all')
        query_dataset = dataset_cls(config['dataset']['root'],
                                    transform=te_transform,
                                    mode='query')
        gallery_dataset = dataset_cls(config['dataset']['root'],
                                    transform=te_transform,
                                    mode='gallery')

        # Combine extra datasets
        for dataset_name in config['dataset']['extras']:
            dataset_cls = get_dataset_cls(dataset_name)
            dataset = dataset_cls(config['dataset']['root'], transform=tr_transform, mode='all')
            train_dataset = train_dataset + dataset

        print("Training dataset")
        print(train_dataset)

        # Define train/validation dataloader
        # =======================================================
        common_config = { 'num_workers': 4, 'pin_memory': True }
        train_labels = [ sample[1] for sample in train_dataset.data ]
        sampler = BalancedBatchSampler(train_labels,
                                    P=config['dataloader']['P'],
                                    K=config['dataloader']['K'])
        self.train_loader = DataLoader(dataset=train_dataset,
                                    batch_sampler=sampler,
                                    **common_config)
        self.query_loader = DataLoader(dataset=query_dataset,
                                    batch_size=config['dataloader']['batch_size'],
                                    shuffle=False,
                                    **common_config)
        self.gallery_loader = DataLoader(dataset=gallery_dataset,
                                    batch_size=config['dataloader']['batch_size'],
                                    shuffle=False,
                                    **common_config)

        # Learning objective
        margin = config['loss']['margin']
        selector = RandomNegativeTripletSelector(margin=margin)
        self.triplet_loss = OnlineTripletLoss(margin, selector)
        self.crossentropy_loss = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=config['optimizer']['lr'], weight_decay=5e-4)
        self.schedular = lr_scheduler.StepLR(self.optimizer, step_size=config['schedular']['step_size'], gamma=0.1)

        # Tensorboard Writer
        # ======================================================
        dataset_name = "_".join([config['dataset']['name']] + config['dataset']['extras'])
        self.log_dir = osp.join(config['train']['log_dir'],
                                config['train']['exp_name'])
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Current state
        self.best_mAP = 0
        self.current_epoch = -1

        # Resume training
        # =======================================================
        if config['train']['resume']:
            checkpoint_path = osp.join(self.log_dir, 'best.pth')
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.schedular.load_state_dict(checkpoint['schedular'])
            self.best_mAP = checkpoint['current_mAP']
            self.current_epoch = checkpoint_path['current_epoch']
            print("Resume training at epoch '{}'".format(self.current_epoch))

    def train(self):
        for epoch in range(self.current_epoch+1, self.config['train']['n_epochs']):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.evaluate()
            self.schedular.step()

    def train_one_epoch(self):
        n_samples = 0
        running_triplet_loss = 0
        running_triplet_count = 0
        running_crossentropy_loss = 0
        running_corrects = 0

        self.model.train()
        for batch_idx, (imgs, pids, _, _) in enumerate(self.train_loader):
            batch_size = len(imgs)
            n_samples += batch_size

            # Dataset
            imgs = imgs.to(self.device)
            pids = pids.to(self.device)

            # Forward & Backward
            self.optimizer.zero_grad()

            embeddings, labels = self.model(imgs)
            triplet_loss, triplet_count = self.triplet_loss(embeddings, pids)
            crossentropy_loss = self.crossentropy_loss(labels, pids)
            loss = triplet_loss + crossentropy_loss
            loss.backward()

            self.optimizer.step()

            # Log result
            running_triplet_loss += (triplet_loss.item()*triplet_count)
            running_triplet_count += triplet_count
            running_crossentropy_loss += crossentropy_loss.item()*batch_size

            preds = torch.max(labels.data, 1)[1]
            corrects = float(torch.sum(preds == pids.data))
            running_corrects += corrects

        # Export training result
        epoch_triplet_loss = running_triplet_loss / running_triplet_count
        epoch_crossentropy_loss = running_crossentropy_loss / n_samples
        epoch_loss = epoch_triplet_loss + epoch_crossentropy_loss
        self.writer.add_scalar("Epoch Train Loss", epoch_loss, self.current_epoch)

        print("Epoch {}:{}, Triplet Loss: {:.5f}, Avg Count: {}, CrossEntropyLoss: {:.5f}, Avg Acc: {:.2f}".format(
                self.current_epoch, self.config['train']['n_epochs'],
                epoch_triplet_loss, running_triplet_count//len(self.train_loader),
                epoch_crossentropy_loss, running_corrects/n_samples))

    def evaluate(self):
        self.model.eval()

        # Query set
        # ====================================
        query_features, query_pids, query_camids = [], [], []
        for imgs, pids, camids, _ in self.query_loader:

            # Compute feature vectors
            imgs = imgs.to(self.device)
            features = self.model(imgs)

            # Convert to numpy array
            features = features.detach().cpu().numpy()
            pids = pids.detach().cpu().numpy()
            camids = camids.detach().cpu().numpy()

            # Keey the result
            query_features.append(features)
            query_pids.append(pids)
            query_camids.append(camids)

        # Concatenate all query result
        query_features = np.concatenate(query_features)
        query_pids = np.concatenate(query_pids)
        query_camids = np.concatenate(query_camids)

        # Gallery set
        # ====================================
        gallery_features, gallery_pids, gallery_camids = [], [], []
        for imgs, pids, camids, _ in self.gallery_loader:

            # Compute feature vectors
            imgs = imgs.to(self.device)
            features = self.model(imgs)

            # Convert to numpy array
            features = features.detach().cpu().numpy()
            pids = pids.detach().cpu().numpy()
            camids = camids.detach().cpu().numpy()

            # Keey the result
            gallery_features.append(features)
            gallery_pids.append(pids)
            gallery_camids.append(camids)

        # Concatenate all gallery result
        gallery_features = np.concatenate(gallery_features)
        gallery_pids = np.concatenate(gallery_pids)
        gallery_camids = np.concatenate(gallery_camids)

        # mAP & CMC
        # ======================================
        ap = 0.0
        cmc = np.zeros(len(gallery_features))

        # Evaluate query feature one-by-one
        for i in range(len(query_features)):
            query_feature = query_features[i]
            query_pid = query_pids[i]
            query_camid = query_camids[i]

            scores = gallery_features.dot(query_feature.T)
            sorted_indexes = np.argsort(scores)
            sorted_indexes = sorted_indexes[::-1]

            pid_indexes = np.argwhere(gallery_pids==query_pid)
            camid_indexes = np.argwhere(gallery_camids==query_camid)

            target_indexes = np.setdiff1d(pid_indexes, camid_indexes, assume_unique=True)
            junk_indexes1 = np.argwhere(gallery_pids==-1)
            junk_indexes2 = np.intersect1d(pid_indexes, camid_indexes)
            invalid_indexes = np.append(junk_indexes1, junk_indexes2)

            ret, ap_tmp, cmc_tmp = compute_AP_CMC(sorted_indexes, target_indexes, invalid_indexes)
            if ret:
                ap = ap + ap_tmp
                cmc = cmc + cmc_tmp

        # Average ap score and CMC
        mAP = ap / len(query_features)
        CMC = cmc / len(query_features)
        print("Epoch {}:{}, Rank@1: {:.2f}, Rank@5: {:.2f}, Rank@10: {:.2f}, mAP: {:.2f}".format(
            self.current_epoch, self.config['train']['n_epochs'],
            CMC[0], CMC[4], CMC[9], mAP))
        self.writer.add_scalar("Rank@1", CMC[0], self.current_epoch)
        self.writer.add_scalar("Rank@5", CMC[4], self.current_epoch)
        self.writer.add_scalar("Rank@10", CMC[9], self.current_epoch)
        self.writer.add_scalar("mAP", mAP, self.current_epoch)

        if self.best_mAP < mAP:
            self.best_mAP = mAP
            self._save_checkpoint()

    def finalize(self):
        pass

    def _save_checkpoint(self):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_epoch': self.current_epoch,
            'current_mAP': self.best_mAP
            }
        checkpoint_path = osp.join(self.log_dir, 'best.pth')
        torch.save(checkpoint_path, checkpoint)
        print("Save checkpoint to '{}'".format(checkpoint_path))
