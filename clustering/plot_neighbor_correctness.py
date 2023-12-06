import torch
from torchmetrics.functional import pairwise_cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

from utils import load_embeddings

device = torch.device("cuda")

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
    embeddings, labels = load_embeddings()
    embeddings = torch.tensor(embeddings).to(device)
    labels = torch.tensor(labels).to(device)

    n_neighbors, correctness = get_neighbor_correctness(embeddings, labels)
    cnts = torch.sum(labels, dim=0).cpu().numpy()

    # for i, row in enumerate(correctness):
    #     plt.plot(n_neighbors, row, label=f"Cat {i} ({cnts[i].item()} data points)")

    # plt.legend()
    # plt.xlabel("Number of nearest neighbors")
    # plt.ylabel("Percent neighbors correct [%]")
    # plt.ylim([0, 1])
    # plt.show()

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, row in enumerate(correctness):
        ax1.plot(n_neighbors, row, label=f"Cat {i}", color=colors[i])
    ax1.set_xlabel("Number of nearest neighbors")
    ax1.set_ylabel("Percent neighbors correct [%]")
    ax1.set_ylim([0, 1])
    ax1.legend()

    ax2.bar(np.arange(len(cnts)), cnts, color=colors[:len(cnts)])
    ax2.set_xlabel('Cat ID')
    ax2.set_ylabel('Number of pictures')

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
