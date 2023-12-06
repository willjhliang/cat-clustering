import numpy as np
import torch
from torchmetrics.functional import pairwise_cosine_similarity
from sklearn import metrics
from sklearn.manifold import TSNE
from scipy import optimize
import matplotlib.pyplot as plt


def plot_correctness(embeddings, labels_true, labels_pred, match, top=None):    
    embeddings = normalize_embeddings(embeddings)

    if (labels_true.ndim == 2):
        labels_true = np.argmax(labels_true, axis=1)
    if (labels_pred.ndim == 2):
        labels_pred = np.argmax(labels_pred, axis=1)

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    output = tsne.fit_transform(embeddings)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    alpha = 1 if top is None else 0.05

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=labels_true.min(), vmax=labels_true.max())
    axs[0].scatter(output[match][:, 0], output[match][:, 1], marker='o', c=cmap(norm(labels_true[match])), alpha=alpha)
    axs[0].scatter(output[np.invert(match)][:, 0], output[np.invert(match)][:, 1], marker='x', c=cmap(norm(labels_true[np.invert(match)])), alpha=alpha)
    if top is not None:
        axs[0].scatter(output[top][:, 0], output[top][:, 1], marker='*', c=cmap(norm(labels_true[top])), s=128)
    axs[0].set_title("Colored by ground truth")

    axs[1].scatter(output[match][:, 0], output[match][:, 1], marker='o', c=cmap(norm(labels_pred[match])), alpha=alpha)
    axs[1].scatter(output[np.invert(match)][:, 0], output[np.invert(match)][:, 1], marker='x', c=cmap(norm(labels_pred[np.invert(match)])), alpha=alpha)
    if top is not None:
        axs[1].scatter(output[top][:, 0], output[top][:, 1], marker='*', c=cmap(norm(labels_pred[top])), s=128)
    axs[1].set_title("Colored by predicted")

    plt.show()


def plot_confident_correctness(embeddings, labels_true, labels_pred, match):    
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


def plot_embeddings(embeddings, labels):
    embeddings = normalize_embeddings(embeddings)

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    output = tsne.fit_transform(embeddings)

    if (labels.ndim > 1):
        label_colors = np.argmax(labels, axis=1)
    else:
        label_colors = labels
    print(label_colors)
    plt.scatter(output[:, 0], output[:, 1], c=label_colors)
    plt.show()

def load_dino():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    dino.eval()
    dino.to(device)
    return dino

def load_embeddings(filename="../embeddings/cls_tokens.npz"):
    embeddings_file = np.load(filename)
    embeddings = embeddings_file["embeddings"]
    labels = embeddings_file["labels"]
    embeddings_file.close()

    return embeddings, labels

def normalize_embeddings(embeddings):
    row_norms = np.linalg.norm(embeddings, axis=1, ord=2)
    return embeddings / row_norms[:, np.newaxis]


def get_neighbors(embeddings, n_neighbors):
    similarities = pairwise_cosine_similarity(embeddings)
    neighbors = np.empty((embeddings.shape[0], n_neighbors))
    for i, row in enumerate(similarities):
        neighbors[i] = torch.topk(row, k=n_neighbors).indices.cpu().numpy()
    return neighbors.astype(int)

def save_model(model):
    torch.save(model.state_dict(), f"model.pt")

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))

def save_output(embeddings, preds, labels):
    np.savez("output.npz", embeddings=embeddings, preds=preds, labels=labels)

def load_output(filename):
    return np.load(filename)

def to_onehot(labels):
    onehot = np.zeros((labels.size, labels.max() + 1))
    onehot[np.arange(labels.size), labels] = 1
    return onehot