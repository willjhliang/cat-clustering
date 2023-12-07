
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
    plt.scatter(output[:, 0], output[:, 1], c=label_colors)
    plt.show()

if __name__ == "__main__":
    embeddings = load_embeddings("cls_token")
    embeddings, labels, img_paths = embeddings["embeddings"], embeddings["labels"], embeddings["img_paths"]

    plot_embeddings(embeddings, labels)