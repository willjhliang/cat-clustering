import gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import hydra
from tqdm import tqdm

from core import ClusterModel, ClusterPatchModel, SCANLoss, EmbeddingDataset
from utils import load_embeddings, load_video_embeddings, save_model, save_output
from evaluate_preds import evaluate_preds

from hydra.utils import get_original_cwd, to_absolute_path

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)


def train_clustering(model, dataloader, optimizer, criterion, labels, epochs):
    model.train()
    for epoch in tqdm(range(epochs+1)):
        epoch_losses = []

        for batch_idx, (embeddings, neighbors, embedding_labels, neighbor_labels) in enumerate(dataloader):
            embeddings = embeddings.to(device)
            neighbors = neighbors.to(device)
            embedding_labels = embedding_labels.to(device)
            neighbor_labels = neighbor_labels.to(device)

            optimizer.zero_grad()
            embeddings_pred = model(embeddings)
            neighbors_pred = model(neighbors)

            if isinstance(criterion, SCANLoss):
                loss, _, _ = criterion(embeddings_pred, neighbors_pred)
            elif isinstance(criterion, nn.CrossEntropyLoss):
                loss = criterion(embeddings_pred, embedding_labels.float())
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        if epoch % 25 == 0:
            print(f"Epoch {epoch}")
            print(f"Loss: {np.mean(epoch_losses)}")
            pass

    return model

@hydra.main(config_path="../config", config_name="config", version_base=None)
def run(cfg):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    n_clusters = cfg.n_clusters
    n_estimated_neighbors = cfg.n_estimated_neighbors
    epochs = cfg.epochs
    lr = cfg.lr
    batch_size = cfg.batch_size
    loss = cfg.loss
    embedding_type = cfg.embedding_type

    embeddings, labels, _ = load_embeddings(embedding_type)

    model = None
    if "cls" in embedding_type:
        model = ClusterModel(n_clusters=n_clusters).to(device)
    elif "patch" in embedding_type:
        model = ClusterPatchModel(n_clusters=n_clusters).to(device)
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = EmbeddingDataset(embedding_type, embeddings, labels, n_estimated_neighbors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if loss == "scan":
        criterion = SCANLoss(cfg.scan_entropy_weight)
    elif loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    model = train_clustering(model, dataloader, optimizer, criterion, labels, epochs=epochs)
    save_model(model, save_dir=output_dir)

    preds = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for _, (cur_embeddings, _, cur_labels, _) in enumerate(dataloader):
        cur_embeddings = cur_embeddings.to(device)
        preds.append(model(cur_embeddings).detach().cpu())


    preds = np.concatenate(preds, axis=0)
    save_output(preds, save_dir=output_dir)

    evaluate_preds(embeddings, labels, preds)

if __name__ == "__main__":
    run()
