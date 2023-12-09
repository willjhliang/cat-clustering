import os
from glob import glob
import torch
from torchvision import transforms as T
from PIL import Image, ExifTags
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_embeddings, load_image

def plot_patches():
    embeddings = load_embeddings("patch_tokens")
    masked_patches, masks, all_patches, all_labels, paths = embeddings["embeddings"], embeddings["masks"], embeddings["unmasked_embeddings"], embeddings["labels"], embeddings["img_paths"]

    # Convert all_patches to RGB for visualization
    features = all_patches
    flattened_features = np.reshape(features, (-1, 1024))

    pca = PCA(n_components=3)
    pca.fit(flattened_features)
    pca_features = pca.transform(flattened_features)
    pca_features = np.reshape(pca_features, (-1, 256, 3))
    for i in range(pca_features.shape[0]):
        pca_features[i, :, :] = (pca_features[i, :, :] - pca_features[i, :, :].min()) / (pca_features[i, :, :].max() - pca_features[i, :, :].min())
    
    all_patches = pca_features

    for i, path in enumerate(paths):
        cur_masked_patches = masked_patches[i]
        cur_all_patches = all_patches[i]
        cur_pca_features = pca_features[i]
        cur_mask = masks[i].reshape(16, 16)[:, :, None]

        # Plot figure
        fig = plt.figure()

        # fig.add_subplot(1, 2, 1)
        # cur_masked_patches = cur_masked_patches.reshape(16, 16, 1024)
        # plt.imshow((cur_masked_patches * 255).astype(np.uint8))

        # fig.add_subplot(1, 2, 2)
        # cur_all_patches = cur_all_patches.reshape(16, 16, 64)
        # plt.imshow((cur_all_patches * 255).astype(np.uint8))

        fig.add_subplot(1, 2, 1)
        cur_pca_features = cur_pca_features.reshape(16, 16, 3)
        cur_pca_features *= cur_mask
        plt.imshow((cur_pca_features * 255).astype(np.uint8))

        fig.add_subplot(1, 2, 2)
        img = load_image(path, normalize=False).permute(1, 2, 0).detach().cpu().numpy()
        img_mask = cur_mask.repeat(14, axis=0).repeat(14, axis=1)
        img *= img_mask
        plt.imshow((img * 255).astype(np.uint8))

        plt.show()

if __name__ == "__main__":
    plot_patches()