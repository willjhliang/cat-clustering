
import torch
import torch.nn as nn
import numpy as np

from utils import get_neighbors

class ClusterModel(nn.Module):
    def __init__(self, n_clusters):
        super(ClusterModel, self).__init__()

        self.linear = nn.Linear(1024, n_clusters)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels, n_neighbors=10):
        super(EmbeddingDataset, self).__init__()

        self.embeddings = torch.tensor(embeddings)
        self.labels = torch.tensor(labels)
        self.n_neighbors = 10
        self.neighbor_indices = get_neighbors(self.embeddings, self.n_neighbors)

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        neighbor_idx = np.random.choice(self.neighbor_indices[idx])

        embedding = self.embeddings[idx]
        neighbor = self.embeddings[neighbor_idx]
        embedding_label = self.labels[idx]
        neighbor_label = self.labels[neighbor_idx]

        return embedding, neighbor, embedding_label, neighbor_label


class SCANLoss(nn.Module):
    """SCAN: Learning to Classify Images without Labels, ECCV 2020"""

    def __init__(self, entropy_weight = 2.0):
        super(SCANLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0
    
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
       
        # Similarity in output space
        similarity = torch.bmm(anchors.view(b, 1, n), neighbors.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        
        # Entropy loss
        entropy_loss = self.entropy(torch.mean(anchors, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        
        return total_loss, consistency_loss, entropy_loss
