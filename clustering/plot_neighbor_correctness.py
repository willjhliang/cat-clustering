import torch
from torchmetrics.functional import pairwise_cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

from utils import load_embeddings

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)

def get_neighbor_correctness(embeddings, labels, max_neighbors=50):
    similarities = pairwise_cosine_similarity(embeddings)
    n_neighbors = np.arange(1, max_neighbors + 1)
    correctness = np.empty((labels.shape[1], max_neighbors))

    for n in n_neighbors:
        matches = np.zeros(labels.shape[1])
        cnts = np.zeros(labels.shape[1])
        for i, row in enumerate(similarities):
            row_label_vec = labels[i]
            row_label_ind = torch.argmax(labels[i]).item()
            neighbor_inds = torch.topk(row, k=n).indices
            neighbor_labels = torch.sum(labels[neighbor_inds], dim=0)
            num_match = torch.dot(row_label_vec.double(), neighbor_labels.double()).item()
            matches[row_label_ind] += num_match
            cnts[row_label_ind] += 1
        percents = matches / (cnts * n)
        correctness[:,n-1] = percents

    return n_neighbors, correctness

    

def main():
    embeddings, labels, _ = load_embeddings("cls_tokens")
    embeddings = torch.tensor(embeddings).to(device)
    labels = torch.tensor(labels).to(device)

    n_neighbors, correctness = get_neighbor_correctness(embeddings, labels)

    # Create a figure with two subplots
    plt.rcParams["image.cmap"] = "Spectral"

    for i, row in enumerate(correctness):
        plt.plot(n_neighbors, row, label=f"{i}")
    plt.xlabel("Number of nearest neighbors")
    plt.ylabel("Percent neighbors correct [%]")
    plt.ylim([0, 1])
    plt.legend(title="Classes")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
