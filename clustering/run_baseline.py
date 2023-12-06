
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import hydra
from tqdm import tqdm

from core import SCANLoss, EmbeddingDataset
from utils import load_embeddings, normalize_embeddings, get_neighbors, save_model

def get_neighbors(anchor, embeddings, n_neighbors):
    from torchmetrics.functional import pairwise_cosine_similarity
    similarities = pairwise_cosine_similarity(anchor, embeddings)
    # print(anchor.shape)
    # print(embeddings.shape)
    # print("hi")
    # print(similarities.shape)
    neighbors = np.empty((anchor.shape[0], n_neighbors))
    for i, row in enumerate(similarities):
        neighbors[i] = neighbor_inds = torch.topk(row, k=n_neighbors).indices.cpu().numpy()
    return neighbors.astype(int)

class BaselineKNNModel(nn.Module):
    def __init__(self, embeddings, labels, n_neighbors=1):
        super(BaselineKNNModel, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings = torch.tensor(embeddings).to(self.device)
        self.labels = torch.tensor(labels).to(self.device)
        self.n_neighbors = n_neighbors
    
    def forward(self, x):
        print("forward")
        neighbors = torch.tensor(get_neighbors(x, self.embeddings, self.n_neighbors)).to(self.device)
        print(neighbors.shape)
        print(self.labels.shape)
        neighbor_labels = self.labels[neighbors]
        # neighbor_labels = self.labels[neighbors]
        print(neighbor_labels.shape)
        pred = torch.mode(neighbor_labels, dim=1).values
        print(pred.shape)
        print("done")
        return pred

def eval_clustering(model, dataloader, criterion, labels, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    epoch_accuracies = []

    for batch_idx, (embeddings, _, embedding_labels, _) in enumerate(dataloader):
        embeddings = embeddings.to(device)
        labels = embedding_labels.to(device)

        pred = model(embeddings)
        print(pred.shape)
        print(labels.shape)
        print(pred == labels)
        accuracy = torch.sum(pred == labels).item() / len(pred)

        epoch_accuracies.append(accuracy)

    print(f"Accuracy: {np.mean(epoch_accuracies)}")
    print()

@hydra.main(config_path="../config", config_name="config", version_base=None)
def run(cfg):
    n_clusters = cfg.n_clusters
    n_neighbors = cfg.n_neighbors
    epochs = cfg.epochs
    lr = cfg.lr
    batch_size = cfg.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings, labels = load_embeddings()

    model = BaselineKNNModel(embeddings, labels, n_neighbors=1).to(device)
    dataset = EmbeddingDataset(embeddings, labels, n_neighbors=n_neighbors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = SCANLoss()

    model = eval_clustering(model, dataloader, criterion, labels, epochs=epochs)



if __name__ == "__main__":
    run()

# def run_clustering():
#     embeddings, labels = load_embeddings()
#     embeddings = normalize_embeddings(embeddings)

