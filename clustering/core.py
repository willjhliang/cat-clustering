import gc

import torch
import torch.nn as nn
import numpy as np

from utils import get_estimated_neighbors, get_true_neighbors, load_embeddings

class ClusterModel(nn.Module):
    def __init__(self, n_clusters):
        super(ClusterModel, self).__init__()
        self.linear = nn.Linear(1024, n_clusters)
    
    def forward(self, x):
        x = self.linear(x)
        return x

class ClusterPatchModel(nn.Module):
    def __init__(self, n_clusters):
        super(ClusterPatchModel, self).__init__()

        self.model_conv = nn.Sequential(
            nn.Conv2d(1024, 2048, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(2048, 4096, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.model_linear = nn.Sequential(
            nn.Linear(4096, n_clusters),
        )
        
    def forward(self, x):
        x = self.model_conv(x)
        x = x.reshape(x.shape[:2])
        x = self.model_linear(x)
        return x

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embedding_type, embeddings, labels, n_neighbors=10):
        super(EmbeddingDataset, self).__init__()
        self.embedding_type = embedding_type

        self.embeddings = torch.tensor(embeddings)
        self.labels = torch.tensor(labels).int()
        self.n_neighbors = n_neighbors
        # cls_tokens, _, _ = load_embeddings("cls_tokens")
        cls_tokens = embeddings
        self.neighbor_indices = get_estimated_neighbors(torch.tensor(cls_tokens), self.n_neighbors)

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        neighbor_idx = np.random.choice(self.neighbor_indices[idx])

        embedding = self.reshape_embedding(self.embeddings[idx])
        neighbor = self.reshape_embedding(self.embeddings[neighbor_idx])
        embedding_label = self.labels[idx]
        neighbor_label = self.labels[neighbor_idx]

        return embedding, neighbor, embedding_label, neighbor_label
    
    def reshape_embedding(self, embedding):
        if self.embedding_type == "cls_tokens":
            return embedding
        elif self.embedding_type == "zoom_cls_token":
            return embedding
        elif self.embedding_type == "patch_tokens":
            return embedding.reshape(16, 16, 1024).permute(2, 0, 1)
        else:
            return embedding


class SCANLoss(nn.Module):
    """SCAN: Learning to Classify Images without Labels, ECCV 2020"""

    def __init__(self, entropy_weight = 1.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight
    
    def entropy(self, x, input_as_probabilities):
        """ 
        Helper function to compute the entropy over the batch 

        input: batch w/ shape [b, num_classes]
        output: entropy value [is ideally -log(num_classes)]
        """

        if input_as_probabilities:
            x_ =  torch.clamp(x, min = 1e-8)
            b =  x_ * torch.log(x_)
        else:
            b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

        if len(b.size()) == 2: # Sample-wise entropy
            return -b.sum(dim = 1).mean()
        elif len(b.size()) == 1: # Distribution-wise entropy
            return - b.sum()
        else:
            raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))

    def forward(self, anchors, neighbors):
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
       
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        
        # Entropy loss
        entropy_loss = self.entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        
        return total_loss, consistency_loss, entropy_loss
