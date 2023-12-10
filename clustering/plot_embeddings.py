
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse

from utils import normalize_embeddings, load_embeddings

def plot_embeddings(embeddings, labels, save_filename=None):
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    embeddings = normalize_embeddings(embeddings)

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    output = tsne.fit_transform(embeddings)

    if (labels.ndim > 1):
        label_colors = np.argmax(labels, axis=1)
    else:
        label_colors = labels
    
    plt.rcParams["image.cmap"] = "Spectral"
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    scatter = ax.scatter(output[:, 0], output[:, 1], c=label_colors)
    legend1 = ax.legend(*scatter.legend_elements(num=None), loc="upper right", title="Classes")
    ax.add_artist(legend1)
    if save_filename is not None:
        plt.savefig(f"{save_filename}.png")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embedding_type", type=str, required=True, help="embedding type, one of the names in embeddings/")
    parser.add_argument("-s", "--save_filename", type=str, help="filename to save plot to")
    args = parser.parse_args()
    save_filename = args.save_filename
    if save_filename == "":
        save_filename = args.embedding_type

    embeddings, labels, _ = load_embeddings(args.embedding_type)
    plot_embeddings(embeddings, labels, save_filename=save_filename)