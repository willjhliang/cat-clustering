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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(dir, size=None):
    paths = []
    for extension in ["jpg", "jpeg", "png"]:
        for filename in glob(os.path.join(dir, f"**/*.{extension}"), recursive=True):
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

def run_dino(dino, data, batch_size=64):
    num_data = data.shape[0]
    num_batches = (num_data + batch_size - 1) // batch_size

    # initialize an empty array to store the predictions
    all_cls = []
    all_patches = []

    # process the data in batches1
    for batch_index in tqdm(range(num_batches)):
        start = batch_index * batch_size
        end = min((batch_index + 1) * batch_size, num_data)
        batch_inputs = data[start:end].to(device)

        # run predictions on the current batch
        with torch.no_grad():
            batch_cls = dino(batch_inputs).detach().cpu().numpy()
            batch_patches = dino.forward_features(batch_inputs)["x_norm_patchtokens"].detach().cpu().numpy()

        all_cls.append(batch_cls)
        all_patches.append(batch_patches)

    # concatenate the predictions from all batches
    all_cls = np.concatenate(all_cls, axis=0)
    all_patches = np.concatenate(all_patches, axis=0)
    return all_cls, all_patches

def run_dino_batched(dino, data_dir, batch_size=64, data_size=None):
    paths = []
    for extension in ["jpg", "jpeg", "png"]:
        for filename in glob(os.path.join(data_dir, f"**/*.{extension}"), recursive=True):
            paths.append(filename)
    if data_size is not None:
        paths = np.random.choice(paths, data_size, replace=False)
    
    num_data = len(paths)
    num_batches = (num_data + batch_size - 1) // batch_size

    all_cls, all_patches, all_labels = [], [], []
    for batch_index in tqdm(range(num_batches)):
        start = batch_index * batch_size
        end = min((batch_index + 1) * batch_size, num_data)
        batch_paths = paths[start:end]

        imgs = [load_image(path) for path in batch_paths]
        labels = [path.split("/")[-2] for path in batch_paths]
        batch_inputs = torch.stack(imgs).to(device)

        with torch.no_grad():
            batch_cls = dino(batch_inputs).detach().cpu().numpy()
            batch_patches = dino.forward_features(batch_inputs)["x_norm_patchtokens"].detach().cpu().numpy()

        all_cls.append(batch_cls)
        all_patches.append(batch_patches)
        all_labels.extend(labels)
    
    unique_categories = list(set(all_labels))
    one_hot_vectors = []
    for label in all_labels:
        one_hot_vector = [1 if label == i else 0 for i in unique_categories]
        one_hot_vectors.append(one_hot_vector)
    all_labels = np.array(one_hot_vectors)

    all_cls = np.concatenate(all_cls, axis=0)
    all_patches = np.concatenate(all_patches, axis=0)
    all_labels = np.array(all_labels)

    return all_cls, all_patches, all_labels, paths


def extract_cls(dino, data_dir, batch_size=64, data_size=None):
    all_cls, _, all_labels, paths = run_dino_batched(dino, data_dir, batch_size=batch_size)
    return all_cls, all_labels, paths

def extract_patches(dino, data_dir, batch_size=64, data_size=None):
    _, all_patches, all_labels, paths = run_dino_batched(dino, data_dir, batch_size=batch_size)
    return all_patches, all_labels, paths

def extract_masked_patches(dino, data_dir, batch_size=64, data_size=None):
    _, all_patches, all_labels, paths = run_dino_batched(dino, data_dir, batch_size=batch_size, data_size=data_size)

    flattened_features = np.reshape(all_patches, (-1, 1024))
    pca = PCA(n_components=3)
    pca.fit(flattened_features)
    pca_features = pca.transform(flattened_features)
    pca_features = np.reshape(pca_features, (-1, 256, 3))
    masks = (pca_features[:, :, 0] - pca_features[:, :, 0].min()) / (pca_features[:, :, 0].max() - pca_features[:, :, 0].min())
    masks = masks[:, :, None]

    masked_patches = all_patches * masks
    return masked_patches, masks, all_patches, all_labels, paths


def main():
    dino = load_dino()

    # inputs, labels, paths = load_data("../data", size=None)

    cls_embeddings, labels, paths = extract_cls(dino, "../data", batch_size=64, data_size=None)
    np.savez("../embeddings/cls_tokens.npz", embeddings=cls_embeddings, labels=labels, img_paths=paths)
    # masked_patches, masks, all_patches, all_labels, paths = extract_masked_patches(dino, "../data", batch_size=64, data_size=None)
    # all_patches, all_labels, paths = extract_patches(dino, "../data", batch_size=64, data_size=None)

    # Code to save embeddings as npy file
    # np.savez("../embeddings/masked_patches.npz", embeddings=masked_patches, masks=masks, unmasked_embeddings=all_patches, labels=all_labels, img_paths=paths)
    # np.savez("../embeddings/patches.npz", embeddings=all_patches, labels=all_labels, img_paths=paths)





    # SCRATCH CODE STARTS HERE

    # dirpath = "../data"
    # for subdir, _, files in os.walk(dirpath):
    #     paths = [os.path.join(subdir, file) for file in files]
    # paths = np.random.choice(paths, 100, replace=False)

    # imgs = [load_image(path) for path in paths]
    # data = torch.stack(imgs)

    # ##################################################################
    # batch_size = 64
    # num_data = data.shape[0]
    # num_batches = (num_data + batch_size - 1) // batch_size

    # # initialize an empty array to store the predictions
    # all_predictions = []

    # # process the data in batches1
    # for batch_index in tqdm(range(num_batches)):
    #     start = batch_index * batch_size
    #     end = min((batch_index + 1) * batch_size, num_data)
    #     batch_inputs = data[start:end].to(device)

    #     # run predictions on the current batch
    #     with torch.no_grad():
    #         # batch_predictions = dino(batch_inputs).detach().cpu().numpy()
    #         batch_predictions = dino.forward_features(batch_inputs)["x_norm_patchtokens"].detach().cpu().numpy()

    #     all_predictions.append(batch_predictions)

    # # concatenate the predictions from all batches
    # predictions = np.concatenate(all_predictions, axis=0)
    # ##################################################################

    # features = predictions
    # flattened_features = np.reshape(features, (-1, 1024))

    # pca = PCA(n_components=3)
    # pca.fit(flattened_features)
    # pca_features = pca.transform(flattened_features)
    # pca_features = np.reshape(pca_features, (-1, 256, 3))

    # pca_features = (pca_features - pca_features[:, :, 0].min()) / (pca_features[:, :, 0].max() - pca_features[:, :, 0].min())

    # for i, path in enumerate(paths):
    #     print(path)
    #     cur_pca_features = pca_features[i]

    #     mask = cur_pca_features[:, 0].reshape(16, 16)
    #     # mask = (mask - mask.min()) / (mask.max() - mask.min())
    #     # mask = cur_pca_features[:, 0] < 0.67
    #     pca_mask = mask
    #     img_mask = mask.repeat(14, axis=0).repeat(14, axis=1)

    #     # For visualization purposes only, to display full RGB spectrum
    #     cur_pca_features = (cur_pca_features - cur_pca_features.min()) / (cur_pca_features.max() - cur_pca_features.min())

    #     # Plot figure
    #     fig = plt.figure()

    #     fig.add_subplot(1, 2, 1)
    #     cur_pca_features = cur_pca_features.reshape(16, 16, 3)
    #     # cur_pca_features[pca_mask, :] = 0
    #     for i in range(3):
    #         cur_pca_features[:, :, i] *= pca_mask
    #     plt.imshow((cur_pca_features * 255).astype(np.uint8))

    #     fig.add_subplot(1, 2, 2)
    #     img = load_image(path, normalize=False).permute(1, 2, 0).detach().cpu().numpy()
    #     # img[:, img_mask] = 0
    #     for i in range(3):
    #         img[:, :, i] *= img_mask

    #     plt.imshow((img * 255).astype(np.uint8))

    #     plt.show()



if __name__ == "__main__":
    main()