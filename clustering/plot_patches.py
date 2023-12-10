import os
from glob import glob
import torch
from torchvision import transforms as T
from PIL import Image, ExifTags
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import argparse

from utils import load_embeddings, load_image

def plot_patches(embedding_type, iteration_scheme):
    patches, labels, paths = load_embeddings(embedding_type)

    # Convert all_patches to RGB via PCA
    flattened_features = np.reshape(patches, (-1, 1024))

    # Remove all-zero patches, add them back after PCA
    zero_mask = np.all(flattened_features == 0, axis=1)
    flattened_features_nonzero = flattened_features[~zero_mask]

    pca = PCA(n_components=3)
    pca.fit(flattened_features_nonzero)
    pca_features_nonzero = pca.transform(flattened_features_nonzero)

    pca_features = np.zeros((flattened_features.shape[0], 3))
    pca_features[~zero_mask] = pca_features_nonzero
    pca_features = np.reshape(pca_features, (-1, 256, 3))

    pca_features_global_norm = pca_features.copy()
    pca_features_local_norm = pca_features.copy()

    # target_paths = [
    #     "../data/thebengalalex/2023-10-01_12-40-13_UTC.jpg",
    #     "../data/thebengalalex/2023-05-04_16-04-25_UTC_2.jpg",
    #     "../data/aslantheforestcat/2019-02-08_15-51-11_UTC.jpg",
    #     "../data/aslantheforestcat/2019-01-30_16-40-22_UTC.jpg",
    #     "../data/cobythecat/2022-01-22_16-18-59_UTC.jpg",
    #     "../data/cobythecat/2022-07-14_14-52-43_UTC_2.jpg",
    #     "../data/fantasticflora/2016-11-05_16-24-18_UTC.jpg",
    #     "../data/fantasticflora/2016-08-25_05-04-14_UTC.jpg",
    #     "../data/russianblueleia/2021-01-17_18-54-39_UTC.jpg",
    #     "../data/russianblueleia/2020-10-09_15-50-42_UTC.jpg",
    #     "../data/lemon/IMG_7983.jpeg",
    #     "../data/lemon/IMG_7142.jpeg",
    #     "../data/luffylittleleopard/2020-11-03_02-30-42_UTC.jpg",
    #     "../data/luffylittleleopard/2022-08-07_14-19-47_UTC.jpg",
    #     "../data/manduthebengalcat/2022-08-22_14-11-31_UTC_1.jpg",
    #     "../data/manduthebengalcat/2023-04-06_16-03-20_UTC_1.jpg",
    #     "../data/mocha/IMG_7865.jpeg",
    #     "../data/mocha/IMG_0174.jpeg",
    #     "../data/nelsonnthecat/2019-01-06_20-11-42_UTC.jpg",
    #     "../data/nelsonnthecat/2019-01-01_20-01-07_UTC_1.jpg",
    #     "../data/olivia_bengal/2021-07-20_17-09-24_UTC.jpg",
    #     "../data/olivia_bengal/2020-05-17_17-44-20_UTC_2.jpg",
    #     "../data/raythebengalcat/2023-03-22_16-12-37_UTC_1.jpg",
    #     "../data/raythebengalcat/2023-04-11_14-28-55_UTC.jpg",
    #     "../data/sterlingsilvercat/2019-09-16_02-05-58_UTC.jpg",
    #     "../data/sterlingsilvercat/2020-08-01_17-27-36_UTC.jpg",
    #     "../data/superjoep/2017-11-15_14-43-23_UTC.jpg",
    #     "../data/superjoep/2020-02-27_15-03-25_UTC.jpg",
    #     "../data/toki_the_bengal/2022-01-04_17-40-55_UTC.jpg",
    #     "../data/toki_the_bengal/2022-03-15_16-41-08_UTC.jpg",
    #     "../data/umasiberian/2022-04-22_19-17-46_UTC_2.jpg",
    #     "../data/umasiberian/2023-02-10_18-13-56_UTC.jpg"
    # ]

    # Normalize PCA features across all images
    for j in range(3):
        cur_pca_features = pca_features_global_norm[:, :, j]
        nonzero_mask = cur_pca_features != 0
        min_val = np.min(cur_pca_features[nonzero_mask])
        max_val = np.max(cur_pca_features[nonzero_mask])
        cur_pca_features = (cur_pca_features - min_val) / (max_val - min_val)
        cur_pca_features[~nonzero_mask] = 0
        pca_features_global_norm[:, :, j] = cur_pca_features

    # Normalize PCA features per image
    for i in range(pca_features_local_norm.shape[0]):
        cur_pca_features = pca_features_local_norm[i]
        for j in range(3):
            nonzero_mask = cur_pca_features[:, j] != 0
            min_val = np.min(cur_pca_features[nonzero_mask, j])
            max_val = np.max(cur_pca_features[nonzero_mask, j])
            cur_pca_features[:, j] = (cur_pca_features[:, j] - min_val) / (max_val - min_val)
        cur_pca_features[~nonzero_mask] = 0
        pca_features_local_norm[i] = cur_pca_features
    
    class_names = sorted(list(set([path.split("/")[-2] for path in paths])))
    class_indices = {}
    for i in range(len(paths)):
        class_name = paths[i].split("/")[-2]
        key = class_names.index(class_name)
        if key not in class_indices.keys():
            class_indices[key] = []
        class_indices[key].append(i)
    
    i = -1
    class_idx = -1
    while True:
        if iteration_scheme == "random":
            i = np.random.randint(0, len(patches))
        elif iteration_scheme == "sequential":
            i = (i + 1) % len(patches)
        elif iteration_scheme == "class":
            class_idx = (class_idx + 1) % len(class_indices.keys())
            i = np.random.choice(class_indices[class_idx])
        else:
            raise ValueError(f"Invalid iteration scheme: {iteration_scheme}")
        
        path = paths[i]
        # if len(target_paths) == 0:
        #     break
        # if path not in target_paths:
        #     continue
        # else:
        #     target_paths.remove(path)

        cur_pca_features_global_norm = pca_features_global_norm[i]
        cur_pca_features_local_norm = pca_features_local_norm[i]

        fig, axs = plt.subplots(1, 3, figsize=(10, 4))

        cur_pca_features_global_norm = cur_pca_features_global_norm.reshape(16, 16, 3)
        axs[0].imshow((cur_pca_features_global_norm * 255).astype(np.uint8))
        axs[0].set_title("PCA (global norm)")

        cur_pca_features_local_norm = cur_pca_features_local_norm.reshape(16, 16, 3)
        axs[1].imshow((cur_pca_features_local_norm * 255).astype(np.uint8))
        axs[1].set_title("PCA (local norm)")

        img = load_image(path, normalize=False).permute(1, 2, 0).detach().cpu().numpy()
        axs[2].imshow((img * 255).astype(np.uint8))
        axs[2].set_title("Original image")

        fig.suptitle(path)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embedding_type", type=str, required=True, help="embedding type, one of the names in embeddings/")
    parser.add_argument("-i", "--iteration_scheme", type=str, default="random", help="iteration scheme, one of [random, sequential, class]")
    args = parser.parse_args()

    plot_patches(args.embedding_type, args.iteration_scheme)