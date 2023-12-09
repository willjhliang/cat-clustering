import numpy as np
import torch
from torchmetrics.functional import pairwise_cosine_similarity
from torchvision import transforms as T
from sklearn import metrics
from sklearn.manifold import TSNE
from scipy import optimize
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
import os

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)

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

def get_transform_big(normalize=True):
    transform = T.Compose([
        T.Resize(896),
        T.CenterCrop(896),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if normalize else lambda x: x
    ])
    return transform

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
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    dino.eval()
    dino.to(device)
    return dino

def load_embeddings(embedding_type):
    filename = f"../embeddings/{embedding_type}.npz"
    embeddings = np.load(filename)
    return embeddings["embeddings"], embeddings["labels"], embeddings["img_paths"]

def load_video_embeddings():
    filename = "../embeddings/data_videos/cls_tokens.npz"
    embeddings = np.load(filename)
    return embeddings["embeddings"], embeddings["labels"], embeddings["img_paths"]

def load_predictions(filename):
    return np.load(filename)

def normalize_embeddings(embeddings):
    row_norms = np.linalg.norm(embeddings, axis=1, ord=2)
    embeddings = embeddings[row_norms != 0]  # Filter out zero embeddings (from hard mask)
    row_norms = row_norms[row_norms != 0]
    return embeddings / row_norms[:, np.newaxis]

def get_estimated_neighbors(embeddings, n_neighbors):
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    similarities = pairwise_cosine_similarity(embeddings)
    neighbors = np.empty((embeddings.shape[0], n_neighbors))
    for i, row in enumerate(similarities):
        neighbors[i] = torch.topk(row, k=n_neighbors).indices.cpu().numpy()
    return neighbors.astype(int)

def get_true_neighbors(labels, n_neighbors):
    neighbors = []
    for i in range(labels.shape[0]):
        same_cluster = np.where(labels == labels[i])[0]
        neighbors.append(np.random.choice(same_cluster, size=n_neighbors, replace=True))
    neighbors = np.array(neighbors)
    return neighbors

def save_model(model, save_dir=""):
    torch.save(model.state_dict(), f"{save_dir}/model.pt")

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))

def save_output(preds, save_dir="", filename="predictions"):
    np.save(f"{save_dir}/{filename}.npy", preds)

def load_image(path, normalize=True):
    transform = get_transform(normalize=normalize)
    img = open_image(path)
    img_tensor = transform(img)

    return img_tensor

def load_image_big(path, normalize=True):
    transform = get_transform_big(normalize=normalize)
    img = open_image(path)
    img_tensor = transform(img)

    return img_tensor

def save_img(img, filepath=""):
    if isinstance(img, np.ndarray):
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)

    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    img.save(filepath)

def to_onehot(labels):
    onehot = np.zeros((labels.size, labels.max() + 1))
    onehot[np.arange(labels.size), labels] = 1
    return onehot