import torch
from torchmetrics.functional import pairwise_cosine_similarity
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import load_embeddings

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)

def get_neighbor_correctness(embeddings, labels, max_neighbors=50):
    similarities = pairwise_cosine_similarity(embeddings)
    n_neighbors = np.arange(1, max_neighbors + 1)
    correctness = np.empty((labels.shape[1], max_neighbors))

    from tqdm import tqdm
    for n in tqdm(n_neighbors):
        matches = np.zeros(labels.shape[1])
        cnts = np.zeros(labels.shape[1])
        for i, row in enumerate(similarities):
            row_label_vec = labels[i]
            row_label_ind = torch.argmax(labels[i]).item()
            neighbor_inds = torch.topk(row, k=n).indices
            neighbor_labels = torch.sum(labels[neighbor_inds], dim=0)
            num_match = torch.dot(row_label_vec.float(), neighbor_labels.float()).item()
            matches[row_label_ind] += num_match
            cnts[row_label_ind] += 1
        percents = matches / (cnts * n)
        correctness[:,n-1] = percents

    return n_neighbors, correctness

def main():
    embeddings, labels, _ = load_embeddings("cls_tokens")
    embeddings = torch.tensor(embeddings).to(device)
    labels = torch.tensor(labels).int().to(device)

    n_neighbors, correctness = get_neighbor_correctness(embeddings, labels)
    cnts = torch.sum(labels, dim=0).cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = matplotlib.colormaps["Spectral"](np.linspace(0, 1, labels.shape[1]))
    for i, row in enumerate(correctness):
        ax1.plot(n_neighbors, row, label=f"Cat {i}", color=colors[i])
    ax1.set_xlabel("Number of nearest neighbors")
    ax1.set_ylabel("Percent neighbors correct [%]")
    ax1.set_ylim([0, 1])
    ax1.legend()

    ax2.bar(np.arange(len(cnts)), cnts, color=colors[:len(cnts)])
    ax2.set_xlabel('Cat ID')
    ax2.set_ylabel('Number of pictures')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
