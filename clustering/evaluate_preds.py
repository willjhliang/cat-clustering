import numpy as np
from sklearn import metrics
from scipy import optimize
import sys
import argparse

from utils import plot_correctness, load_embeddings, load_predictions

def evaluate_preds(embeddings, labels, preds):
    labels_arg, preds_arg = np.argmax(labels, axis=1), np.argmax(preds, axis=1)
    # top = np.argpartition(np.max(preds, axis=1), -32)[-32:] if np.max(preds, axis=1).min() != 1 else None
    
    # Random baseline
    # print(preds.shape)
    # preds_arg = np.random.random_sample(preds_arg.shape)
    # preds_arg = preds_arg * (preds.shape[1])
    # print(preds_arg)
    # preds_arg = preds_arg.astype(int)
    # print(preds_arg)

    cost = metrics.confusion_matrix(preds_arg, labels_arg)
    _, assignments = optimize.linear_sum_assignment(cost, maximize=True)
    swapped_preds = np.empty(preds.shape)
    for i in range(len(preds_arg)):
        preds_arg[i] = assignments[preds_arg[i]]
        for j in range(len(assignments)):
            swapped_preds[i][assignments[j]] = preds[i][j]
    preds = swapped_preds
    match = labels_arg == preds_arg

    # Filter for bengals
    # bengal_cats = [4, 5, 7, 8, 9, 13, 14]
    # row_mask = np.zeros(labels.shape[0], dtype=bool)
    # for i in bengal_cats:
    #     row_mask = np.logical_or(row_mask, labels[:, i] == 1)
    # embeddings = embeddings[row_mask]    
    # labels = np.delete(labels[row_mask], [0, 1, 2, 3, 6, 10, 11, 12, 15], axis=1)
    # preds = np.delete(preds[row_mask], [0, 1, 2, 3, 6, 10, 11, 12, 15], axis=1)
    # labels_arg, preds_arg = np.argmax(labels, axis=1), np.argmax(preds, axis=1)
    # match = labels_arg == preds_arg

    # Filter for species
    # cat_to_species = [3, 6, 2, 4, 0, 0, 4, 0, 0, 0, 5, 7, 1, 0, 0, 2]
    # new_labels = np.zeros((labels.shape[0], max(cat_to_species)+1))
    # for i in range(labels.shape[0]):
    #     cat = np.argmax(labels[i])
    #     new_labels[i, cat_to_species[cat]] = 1
    # labels = new_labels
    # new_preds = np.zeros((preds.shape[0], max(cat_to_species)+1))
    # for i in range(preds.shape[0]):
    #     cat = np.argmax(preds[i])
    #     new_preds[i, cat_to_species[cat]] = 1
    # preds = new_preds
    # labels_arg, preds_arg = np.argmax(labels, axis=1), np.argmax(preds, axis=1)
    # match = labels_arg == preds_arg

    total = 0
    n_clusters = preds_arg.max() + 1
    for i in range(n_clusters):
        true_mask = labels_arg == i
        total += preds_arg[true_mask & match].shape[0] / np.sum(true_mask) / n_clusters

    print(f"Accuracy (normalized): {total}")
    print(f"Accuracy (raw): {np.sum(match) / match.shape[0]}")

    plot_correctness(embeddings, labels_arg, preds_arg, match)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embedding_type", type=str, required=True, help="embedding type, one of the names in embeddings/")
    parser.add_argument("-f", "--filename", type=str, required=True, help="filename for cluster predictions")
    args = parser.parse_args()

    embeddings, labels, _ = load_embeddings(args.embedding_type)
    preds = load_predictions(args.filename)
    
    # preds, labels = np.load(f"{output_filename}/predictions.npy"), np.load(f"{output_filename}/labels.npy")
    evaluate_preds(embeddings, labels, preds)