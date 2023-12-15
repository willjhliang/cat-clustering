import os
from glob import glob
import torch
from torchvision import transforms as T
from PIL import Image, ExifTags
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import *

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)

def load_data(data_dir, size=None):
    paths = []
    for extension in ["jpg", "jpeg", "png"]:
        for filename in glob(os.path.join(data_dir, f"**/*.{extension}"), recursive=True):
            paths.append(filename)
    if size is not None:
        paths = np.random.choice(paths, size, replace=False)

    data = []
    categories = []
    for filename in paths:
        img_tensor = load_image(filename)
        cat = filename.split("/")[-2]
        data.append(img_tensor)
        categories.append(cat)

    unique_categories = list(set(categories))
    one_hot_vectors = []
    for category in categories:
        one_hot_vector = [1 if category == cat else 0 for cat in unique_categories]
        one_hot_vectors.append(one_hot_vector)
    one_hot = np.array(one_hot_vectors)

    # return data, one_hot
    return torch.stack(data), one_hot, paths

def run_dino_batched(dino, data_dir, batch_size=64, data_size=None):
    paths = []
    for extension in ["jpg", "jpeg", "png"]:
        for filename in glob(os.path.join(data_dir, f"**/*.{extension}"), recursive=True):
            paths.append(filename)
    if data_size is not None:
        paths = np.random.choice(paths, data_size, replace=False)
    
    num_data = len(paths)
    num_batches = (num_data + batch_size - 1) // batch_size

    all_cls, all_intermediate_cls, all_patches, all_labels = [], [], [], []
    for batch_index in tqdm(range(num_batches)):
        start = batch_index * batch_size
        end = min((batch_index + 1) * batch_size, num_data)
        batch_paths = paths[start:end]

        imgs = [load_image(path) for path in batch_paths]
        labels = [path.split("/")[-2] for path in batch_paths]
        batch_inputs = torch.stack(imgs).to(device)

        with torch.no_grad():
            batch_cls = dino(batch_inputs).detach().cpu().numpy()
            batch_intermediate_cls = dino.get_intermediate_layers(batch_inputs, n=dino.n_blocks, return_class_token=True)
            batch_intermediate_cls = [i[1].detach().cpu().numpy() for i in batch_intermediate_cls][:-1]
            batch_intermediate_cls = np.stack(batch_intermediate_cls, axis=1)
            batch_patches = dino.forward_features(batch_inputs)["x_norm_patchtokens"].detach().cpu().numpy()
            
        all_cls.append(batch_cls)
        all_intermediate_cls.append(batch_intermediate_cls)
        all_patches.append(batch_patches)
        all_labels.extend(labels)

    # Convert labels from list of strings to one-hot vectors
    unique_categories = sorted(list(set(all_labels)))
    all_labels = np.array([unique_categories.index(label) for label in all_labels])
    all_labels = to_onehot(all_labels)

    all_cls = np.concatenate(all_cls, axis=0)
    all_intermediate_cls = np.concatenate(all_intermediate_cls, axis=0)
    all_patches = np.concatenate(all_patches, axis=0)
    all_labels = np.array(all_labels)

    return all_cls, all_intermediate_cls, all_patches, all_labels, paths

def extract_masked_patches():
    all_patches, all_labels, paths = load_embeddings("patch_tokens")
    flattened_features = np.reshape(all_patches, (-1, 1024))
    pca = PCA(n_components=3)
    pca.fit(flattened_features)
    pca_features = pca.transform(flattened_features)
    pca_features = np.reshape(pca_features, (-1, 256, 3))

    all_masks = pca_features[:, :, 0]
    hard_masks = all_masks > 0.7
    soft_global_masks = all_masks.copy()
    soft_local_masks = all_masks.copy()
    soft_global_masks = (soft_global_masks - soft_global_masks.min()) / (soft_global_masks.max() - soft_global_masks.min())
    for i in range(soft_global_masks.shape[0]):
        soft_global_masks[i] = (soft_global_masks[i] - soft_global_masks[i].min()) / (soft_global_masks[i].max() - soft_global_masks[i].min())

    hard_masked_patches = all_patches * hard_masks[:, :, None]
    soft_global_masked_patches = all_patches * soft_global_masks[:, :, None]
    soft_local_masked_patches = all_patches * soft_local_masks[:, :, None]

    all_zero = np.sum(hard_masked_patches, axis=(1, 2)) == 0
    np.savez("../embeddings/patch_tokens_hard_masked.npz", embeddings=hard_masked_patches[~all_zero], masks=hard_masks[~all_zero], labels=all_labels[~all_zero], img_paths=paths[~all_zero])
    np.savez("../embeddings/patch_tokens_soft_global_masked.npz", embeddings=soft_global_masked_patches, masks=soft_global_masks, labels=all_labels, img_paths=paths)
    np.savez("../embeddings/patch_tokens_soft_local_masked.npz", embeddings=soft_local_masked_patches, masks=soft_local_masks, labels=all_labels, img_paths=paths)

    return hard_masked_patches, soft_global_masked_patches, soft_local_masked_patches, hard_masks, soft_global_masks, soft_local_masks, all_labels, paths


if __name__ == "__main__":
    dino = load_dino()
    all_cls, all_intermediate_cls, all_patches, all_labels, paths = run_dino_batched(dino, "../data", batch_size=64, data_size=None)
    np.savez("../embeddings/cls_tokens.npz", embeddings=all_cls, labels=all_labels, img_paths=paths)
    for i in range(all_intermediate_cls.shape[1]):
        np.savez(f"../embeddings/cls_tokens_intermediate_{i}.npz", embeddings=all_intermediate_cls[:, i], labels=all_labels, img_paths=paths)
    np.savez("../embeddings/patch_tokens.npz", embeddings=all_patches, labels=all_labels, img_paths=paths)

    extract_masked_patches()
