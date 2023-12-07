import gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import hydra
from tqdm import tqdm

from core import ClusterModel, ClusterPatchModel, SCANLoss, EmbeddingDataset
from utils import load_embeddings, normalize_embeddings, get_neighbors, save_model, save_output

from hydra.utils import get_original_cwd, to_absolute_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_clustering(model, dataloader, optimizer, criterion, labels, epochs):
    """
    TODO IDEA!!!!!!!!!!!!!

    SCAN has two stages: unsupervised clustering with SCANLoss (which we're doing below)
    and then fine-tuning with self-labeling, which trains the model to sharpen its confident predictions

    What if we alternate these steps? Our novel contribution???
    """

    model.train()
    for epoch in range(epochs+1):
        epoch_losses = []
        epoch_accuracies = []
        epoch_precisions = []
        epoch_recalls = []

        for batch_idx, (embeddings, neighbors, embedding_labels, neighbor_labels) in enumerate(dataloader):
            embeddings = embeddings.to(device)
            neighbors = neighbors.to(device)
            embedding_labels = embedding_labels.to(device)
            neighbor_labels = neighbor_labels.to(device)

            optimizer.zero_grad()
            embeddings_pred = model(embeddings)
            neighbors_pred = model(neighbors)

            loss, _, _ = criterion(embeddings_pred, neighbors_pred)
            loss.backward()
            optimizer.step()

            same_cluster_pred = torch.argmax(embeddings_pred, dim=1) == torch.argmax(neighbors_pred, dim=1)
            same_cluster_true = torch.argmax(embedding_labels, dim=1) == torch.argmax(neighbor_labels, dim=1)
            accuracy = torch.sum(same_cluster_pred == same_cluster_true).item() / len(same_cluster_pred)
            precision = torch.sum(same_cluster_pred & same_cluster_true).item() / torch.sum(same_cluster_pred).item()
            recall = torch.sum(same_cluster_pred & same_cluster_true).item() / torch.sum(same_cluster_true).item()

            epoch_losses.append(loss.item())
            epoch_accuracies.append(accuracy)
            epoch_precisions.append(precision)
            epoch_recalls.append(recall)

        print(f"Epoch {epoch}")
        print(f"Loss: {np.mean(epoch_losses)}")
        print(f"Accuracy: {np.mean(epoch_accuracies)}")
        print(f"Precision: {np.mean(epoch_precisions)}")
        print(f"Recall: {np.mean(epoch_recalls)}")
        print()

    return model

def f(a,N):
    mask = np.empty(a.size,bool)
    mask[:N] = True
    np.not_equal(a[N:],a[:-N],out=mask[N:])
    return mask

@hydra.main(config_path="../config", config_name="config", version_base=None)
def run(cfg):
    n_clusters = cfg.n_clusters
    n_neighbors = cfg.n_neighbors
    epochs = cfg.epochs
    lr = cfg.lr
    batch_size = cfg.batch_size
    embedding_type = cfg.embedding_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings, labels = load_embeddings(embedding_type)
    labels = labels.astype(np.int8)
    cls_token, _ = load_embeddings("cls_token")
    del _
    gc.collect()
    print(cls_token.shape)
    # embeddings, labels = embeddings[:512], labels[:512]

    # mask = f(np.argmax(labels, axis=1), 200)
    # embeddings_alt = embeddings[mask]
    # labels_alt = labels[mask]

    model = None
    if embedding_type == "cls_token":
        model = ClusterModel(n_clusters=n_clusters).to(device)
    elif embedding_type == "patch_tokens":
        model = ClusterPatchModel(n_clusters=n_clusters).to(device)
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = EmbeddingDataset(embedding_type, embeddings, labels, cls_token, n_neighbors=n_neighbors)
    del embeddings
    del cls_token
    gc.collect()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = SCANLoss()

    model = train_clustering(model, dataloader, optimizer, criterion, labels, epochs=epochs)
    save_model(model)

    preds = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for _, (cur_embeddings, _, _, _) in enumerate(dataloader):
        cur_embeddings = cur_embeddings.to(device)
        preds.append(model(cur_embeddings).detach().cpu())

    preds = np.concatenate(preds, axis=0)
    save_output(preds)


if __name__ == "__main__":
    run()

# def run_clustering():
#     embeddings, labels = load_embeddings()
#     embeddings = normalize_embeddings(embeddings)

