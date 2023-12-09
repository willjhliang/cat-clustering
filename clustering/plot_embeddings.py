
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse

from utils import normalize_embeddings, load_embeddings

def plot_embeddings(embeddings, labels):
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    embeddings = normalize_embeddings(embeddings)

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    output = tsne.fit_transform(embeddings)

    if (labels.ndim > 1):
        label_colors = np.argmax(labels, axis=1)
    else:
        label_colors = labels
    
    for i in range(np.max(label_colors) + 1):
        print(f"Size of class {i}: {np.sum(label_colors == i)}")
    
    plt.rcParams["image.cmap"] = "Spectral"
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    scatter = ax.scatter(output[:, 0], output[:, 1], c=label_colors)
    legend1 = ax.legend(*scatter.legend_elements(num=None), loc="upper right", title="Classes")
    ax.add_artist(legend1)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embedding_type", type=str, required=True, help="embedding type, one of the names in embeddings/")
    args = parser.parse_args()

    embeddings, labels = load_embeddings(args.embedding_type)
    plot_embeddings(embeddings, labels)