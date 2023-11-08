
import os
from glob import glob
import torch
from torchvision import transforms as T
from PIL import Image 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda")

def load_dino():
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    dino.eval()
    dino.to(device)
    return dino

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

    img = Image.open(path)
    img_tensor = transform(img)
    img_tensor = img_tensor.to(device)
    return img_tensor

def load_data(dir):
    data = []
    categories = []
    for filename in glob(os.path.join(dir, "**/*.jpeg"), recursive=True):
        img_tensor = load_image(os.path.join(dir, filename))
        cat = filename.split("/")[-2]

        data.append(img_tensor)
        categories.append(cat)

    unique_categories = list(set(categories))
    one_hot_vectors = []
    for category in categories:
        one_hot_vector = [1 if category == cat else 0 for cat in unique_categories]
        one_hot_vectors.append(one_hot_vector)
    one_hot = np.array(one_hot_vectors)

    return torch.stack(data), one_hot

def predict(dino, data, batch_size=64):
    num_data = data.shape[0]
    num_batches = (num_data + batch_size - 1) // batch_size

    # Initialize an empty array to store the predictions
    all_predictions = []

    # Process the data in batches
    for batch_index in range(num_batches):
        start = batch_index * batch_size
        end = min((batch_index + 1) * batch_size, num_data)
        batch_inputs = data[start:end]

        # Run predictions on the current batch
        with torch.no_grad():
            batch_predictions = dino(batch_inputs).detach().cpu().numpy()

        all_predictions.append(batch_predictions)

    # Concatenate the predictions from all batches
    predictions = np.concatenate(all_predictions, axis=0)
    return predictions


def compute_tsne(features, labels):
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    output = tsne.fit_transform(features)

    label_colors = np.argmax(labels, axis=1)
    plt.scatter(output[:, 0], output[:, 1], c=label_colors)
    plt.show()


def main():
    dino = load_dino()

    inputs, labels = load_data("../data")
    print(f"Data shape: {inputs.shape}")

    # predictions = dino(inputs).detach().cpu().numpy()
    predictions = predict(dino, inputs)
    compute_tsne(predictions, labels)

    # img = load_image("../data_mini/lemon/IMG_0115.jpeg")
    # img = img.unsqueeze(0)
    # features = dino.forward_features(img)["x_norm_patchtokens"].detach().cpu().numpy()
    # features = features[0]
    # print(features.shape)
    # print(img.shape)
    # print(img.min(), img.max())

    # pca = PCA(n_components=3)
    # pca.fit(features)

    # pca_features = pca.transform(features)
    # pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    # pca_features = pca_features * 255

    # fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(pca_features.reshape(16, 16, 3).astype(np.uint8))
    # fig.add_subplot(1, 2, 2)
    # img = load_image("../data_mini/lemon/IMG_0115.jpeg", normalize=False)
    # img *= 255
    # print(img.min(), img.max())
    # plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    # plt.show()



if __name__ == "__main__":
    main()