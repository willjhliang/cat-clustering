import numpy as np
import torch
from torchmetrics.functional import pairwise_cosine_similarity
from torchvision import transforms as T
from sklearn import metrics
from sklearn.manifold import TSNE
from scipy import optimize
import matplotlib.pyplot as plt
from PIL import Image, ExifTags

def open_image(path):
    image = Image.open(path)

    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        
        exif = image._getexif()
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except:
        pass

    return image

def get_transform(normalize=True):
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if normalize else lambda x: x
    ])
    return transform

def load_image(path, normalize=True):
    transform = get_transform(normalize=normalize)
    img = open_image(path)
    img_tensor = transform(img)

    return img_tensor


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



def load_dino():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    dino.eval()
    dino.to(device)
    return dino

def load_embeddings(embedding_type):
    if embedding_type == "cls_token":
        filename = "/home/danielx/Documents/homework/cis5810/clustering/embeddings/cls_tokens.npz"
    elif embedding_type == "patch_tokens":
        filename = "/home/danielx/Documents/homework/cis5810/clustering/embeddings/patches.npz"
    elif embedding_type == "soft_mask_patch_tokens":
        filename = "/home/danielx/Documents/homework/cis5810/clustering/embeddings/soft_mask_patch_tokens.npz"
    embeddings = np.load(filename)
    return embeddings["embeddings"], embeddings["labels"]

def load_predictions(filename):
    return np.load(filename)

def normalize_embeddings(embeddings):
    row_norms = np.linalg.norm(embeddings, axis=1, ord=2)
    return embeddings / row_norms[:, np.newaxis]

def get_neighbors(embeddings, n_neighbors):
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    similarities = pairwise_cosine_similarity(embeddings)
    neighbors = np.empty((embeddings.shape[0], n_neighbors))
    for i, row in enumerate(similarities):
        neighbors[i] = torch.topk(row, k=n_neighbors).indices.cpu().numpy()
    return neighbors.astype(int)

def save_model(model):
    torch.save(model.state_dict(), f"model.pt")

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))

def save_output(preds):
    np.save("predictions.npy", preds)

def to_onehot(labels):
    onehot = np.zeros((labels.size, labels.max() + 1))
    onehot[np.arange(labels.size), labels] = 1
    return onehot