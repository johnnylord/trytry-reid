import torch
import torch.nn as nn
import torch.nn.functional as F

class OnlineTripletLoss(nn.Module):
    """Triplet loss with associated triplet selector"""
    def __init__(self, margin, triplet_selector):
        super().__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, labels):
        """Compute triplet loss

        Arguments:
            - embeddings (torch.Tensor): embeddings of shape (batch, embedding_dim)
            - labels (torch.LongTensor): target labels shape (batch,)

        Return:
            average triplet loss & and number of triplet samples used for
            computing triplet loss.
        """
        target_embeddings = embeddings.detach().cpu().numpy()
        target_labels = labels.detach().cpu().numpy()
        triplets = self.triplet_selector.get_triplets(target_embeddings, target_labels)
        triplets = torch.LongTensor(triplets)
        triplets = triplets.to(embeddings.device)

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
