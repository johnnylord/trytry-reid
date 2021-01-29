import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet34, resnet18


__all__ = [ "resnet50_reid", "resnet34_reid", "resnet18_reid" ]

class ReIDResnet(nn.Module):
    """ReID model with resnet backbone

    Argument:
        resnet (model): resnet model pretrained on imagenet
        resnet_features (int): size of latent features before fc layer in resnet
        features (int): size of reid latent feature
    """
    def __init__(self, resnet, resnet_features, features, classes):
        super().__init__()

        self.encoder = resnet
        self.embedding = nn.Sequential(nn.Linear(resnet_features, features))
        self.bnneck = nn.Sequential(nn.BatchNorm1d(features))
        self.classifier = nn.Sequential(nn.Linear(features, classes))

    def _encoder_forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self._encoder_forward(x)
        embeddings = self.embedding(x)
        norm_embeddings = F.normalize(self.bnneck(embeddings), p=2, dim=1)

        if not self.training:
            return norm_embeddings

        labels = self.classifier(norm_embeddings)
        return embeddings, labels


def resnet18_reid(features=128, classes=1502):
    resnet = resnet18(pretrained=True)
    resnet.layer4[0].downsample[0].stride = (1, 1)
    resnet.layer4[0].conv1.stride = (1, 1)
    resnet.fc = None
    model = ReIDResnet(resnet, 512, features, classes)
    return model

def resnet34_reid(features=128, classes=1502):
    resnet = resnet34(pretrained=True)
    resnet.layer4[0].downsample[0].stride = (1, 1)
    resnet.layer4[0].conv1.stride = (1, 1)
    resnet.fc = None
    model = ReIDResnet(resnet, 512, features, classes)
    return model

def resnet50_reid(features=128, classes=1502):
    resnet = resnet50(pretrained=True)
    resnet.layer4[0].downsample[0].stride = (1, 1)
    resnet.layer4[0].conv2.stride = (1, 1)
    resnet.fc = None
    model = ReIDResnet(resnet, 2048, features, classes)
    return model


if __name__ == "__main__":
    from torchsummary import summary

    # Instantiate model
    model = resnet50_reid()

    # Random input
    summary(model, (3, 256, 128), device="cpu")
