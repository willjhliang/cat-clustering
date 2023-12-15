
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from evaluate_preds import evaluate_preds
from utils import load_embeddings, load_predictions, normalize_embeddings

def plot_preds(embeddings, labels_true, labels_pred, match, top=None):    
    embeddings = normalize_embeddings(embeddings)

    if (labels_true.ndim == 2):
        labels_true = np.argmax(labels_true, axis=1)
    if (labels_pred.ndim == 2):
        labels_pred = np.argmax(labels_pred, axis=1)

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    output = tsne.fit_transform(embeddings)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    alpha = 1 if top is None else 0.05

    plt.rcParams["image.cmap"] = "Spectral"
    norm = plt.Normalize(vmin=labels_true.min(), vmax=labels_true.max())
    scatter0 = axs[0].scatter(output[match][:, 0], output[match][:, 1], marker='o', c=labels_true[match], alpha=alpha)
    axs[0].scatter(output[np.invert(match)][:, 0], output[np.invert(match)][:, 1], marker='x', c=labels_true[np.invert(match)], alpha=alpha)
    if top is not None:
        axs[0].scatter(output[top][:, 0], output[top][:, 1], marker='*', c=labels_true[top], s=128)
    axs[0].set_title("Colored by ground truth")
    legend0 = axs[0].legend(*scatter0.legend_elements(num=None), loc="upper right", title="Classes")
    axs[0].add_artist(legend0)

    scatter1 = axs[1].scatter(output[match][:, 0], output[match][:, 1], marker='o', c=labels_pred[match], alpha=alpha)
    axs[1].scatter(output[np.invert(match)][:, 0], output[np.invert(match)][:, 1], marker='x', c=labels_pred[np.invert(match)], alpha=alpha)
    if top is not None:
        axs[1].scatter(output[top][:, 0], output[top][:, 1], marker='*', c=labels_pred[top], s=128)
    axs[1].set_title("Colored by predicted")
    legend1 = axs[1].legend(*scatter1.legend_elements(num=None), loc="upper right", title="Classes")
    axs[1].add_artist(legend1)

    plt.show()

def plot_confident_preds(embeddings, labels_true, labels_pred, match):    
    embeddings = normalize_embeddings(embeddings)

    if (labels_true.ndim == 2):
        labels_true = np.argmax(labels_true, axis=1)
    if (labels_pred.ndim == 2):
        labels_pred = np.argmax(labels_pred, axis=1)

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    output = tsne.fit_transform(embeddings)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    cmap = plt.cm.Spectral
    norm = plt.Normalize(vmin=labels_true.min(), vmax=labels_true.max())
    axs[0].scatter(output[match][:, 0], output[match][:, 1], marker='o', c=cmap(norm(labels_true[match])))
    axs[0].scatter(output[np.invert(match)][:, 0], output[np.invert(match)][:, 1], marker='x', c=cmap(norm(labels_true[np.invert(match)])))
    axs[0].set_title("Colored by ground truth")

    axs[1].scatter(output[match][:, 0], output[match][:, 1], marker='o', c=cmap(norm(labels_pred[match])))
    axs[1].scatter(output[np.invert(match)][:, 0], output[np.invert(match)][:, 1], marker='x', c=cmap(norm(labels_pred[np.invert(match)])))
    axs[1].set_title("Colored by predicted")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embedding_type", type=str, required=True, help="embedding type, one of the names in embeddings/")
    parser.add_argument("-f", "--filename", type=str, required=True, help="filename for cluster predictions")
    args = parser.parse_args()

    embeddings, labels, _ = load_embeddings(args.embedding_type)
    preds = load_predictions(args.filename)
    
    labels_arg, preds_arg, match = evaluate_preds(embeddings, labels, preds)

    plot_preds(embeddings, labels_arg, preds_arg, match)