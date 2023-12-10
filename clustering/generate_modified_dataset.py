
import numpy as np


def generate_species_dataset():
    data = np.load("../embeddings/cls_tokens.npz")
    embeddings, labels, paths = data['embeddings'], data['labels'], data['img_paths']

    # Harcoded based on alphabetical species order in README
    cat_to_species = [3, 6, 2, 4, 0, 0, 4, 0, 0, 0, 5, 7, 2, 0, 0, 2]

    new_labels = np.zeros((labels.shape[0], max(cat_to_species)+1))
    for i in range(labels.shape[0]):
        cat = np.argmax(labels[i])
        new_labels[i, cat_to_species[cat]] = 1
    
    np.savez("../embeddings/cls_tokens_species.npz", embeddings=embeddings, labels=new_labels, img_paths=paths)

def generate_bengal_dataset():
    data = np.load("../embeddings/cls_tokens.npz")
    embeddings, labels, paths = data['embeddings'], data['labels'], data['img_paths']

    # Harcoded based on cat list in README
    bengal_cats = [4, 5, 7, 8, 9, 13, 14]
    row_mask = np.zeros(labels.shape[0], dtype=bool)
    for i in bengal_cats:
        row_mask = np.logical_or(row_mask, labels[:, i] == 1)
    
    new_embeddings = embeddings[row_mask]
    new_labels = labels[row_mask]
    new_paths = paths[row_mask]

    new_labels = np.delete(new_labels, [0, 1, 2, 3, 6, 10, 11, 12, 15], axis=1)

    np.savez("../embeddings/cls_tokens_bengal.npz", embeddings=new_embeddings, labels=new_labels, img_paths=new_paths)

if __name__ == "__main__":
    # generate_species_dataset()
    generate_bengal_dataset()

