import os
from glob import glob
import torch
from torchvision import transforms as T
from PIL import Image, ExifTags
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from calculate_embeddings import load_data, run_dino, run_dino_batched
from utils import *

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", action="store_true", help="visualize result instead of saving file")
    parser.add_argument("--global_norm", action="store_true", help="if soft mask, normalize mask globally instead of per image")
    parser.add_argument("--threshold", type=float, default=0.7, help="if hard mask, blackout threshold")
    parser.add_argument("--data_size", type=int, default=0, help="number of images to use")
    args = parser.parse_args()

    dino = load_dino()

    data_size = args.data_size if args.data_size > 0 else None
    all_cls, all_patches, all_labels, paths = run_dino_batched(dino, "../data", data_size=data_size)

    features = all_patches
    flattened_features = np.reshape(features, (-1, 1024))

    pca = PCA(n_components=3)
    pca.fit(flattened_features)
    pca_features = pca.transform(flattened_features)
    pca_features = np.reshape(pca_features, (-1, 256, 3))

    all_masks = pca_features[:, :, 0]
    hard_masks = all_masks > args.threshold
    soft_masks = all_masks
    if args.global_norm:
        soft_masks = (soft_masks - soft_masks.min()) / (soft_masks.max() - soft_masks.min())
    else:
        for i in range(all_masks.shape[0]):
            soft_masks[i] = (all_masks[i] - all_masks[i].min()) / (all_masks[i].max() - all_masks[i].min())

    for i, path in tqdm(enumerate(paths)):
        hard_mask = hard_masks[i].reshape(16, 16)[:, :, None]
        soft_mask = soft_masks[i].reshape(16, 16)[:, :, None]
        img = load_image(path, normalize=False).permute(1, 2, 0).detach().cpu().numpy()
        hard_img = img * hard_mask.repeat(14, axis=0).repeat(14, axis=1)
        soft_img = img * soft_mask.repeat(14, axis=0).repeat(14, axis=1)

        if args.visualize:
            cur_pca_features = pca_features[i]
            # For visualization purposes only, to display full RGB spectrum
            cur_pca_features = (cur_pca_features - cur_pca_features.min()) / (cur_pca_features.max() - cur_pca_features.min())
            cur_pca_features = cur_pca_features.reshape(16, 16, 3)

            fig = plt.figure()
            fig.add_subplot(2, 2, 1)
            plt.imshow((cur_pca_features * soft_mask * 255).astype(np.uint8))
            fig.add_subplot(2, 2, 2)
            plt.imshow((soft_img * 255).astype(np.uint8))
            fig.add_subplot(2, 2, 3)
            plt.imshow((cur_pca_features * hard_mask * 255).astype(np.uint8))
            fig.add_subplot(2, 2, 4)
            plt.imshow((hard_img * 255).astype(np.uint8))
            plt.show()
        else:
            save_img(hard_img, f"../data_masked_hard/{path.split('/')[-2]}/{path.split('/')[-1]}")
            save_img(soft_img, f"../data_masked_soft/{path.split('/')[-2]}/{path.split('/')[-1]}")


if __name__ == "__main__":
    main()