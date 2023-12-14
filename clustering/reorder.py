
import numpy as np

if __name__ == "__main__":
    data = np.load("../embeddings_old/cls_tokens_masked_soft_unordered_classes.npz")
    embeddings, labels, paths = data['embeddings'], data['labels'], data['img_paths']

    classes = sorted(list(set([i.split("/")[-2] for i in paths])))
    new_labels = np.zeros_like(labels)
    for i in range(labels.shape[0]):
        new_labels[i, classes.index(paths[i].split("/")[-2])] = 1
    
    np.savez("../embeddings/cls_tokens_masked_soft.npz", embeddings=embeddings, labels=new_labels, img_paths=paths)