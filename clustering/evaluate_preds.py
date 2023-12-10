import numpy as np
from sklearn import metrics
from scipy import optimize
import sys
import argparse

from utils import plot_correctness, load_embeddings, load_predictions

def evaluate_preds(embeddings, labels, preds):
    labels, preds_arg = np.argmax(labels, axis=1), np.argmax(preds, axis=1)
    top = np.argpartition(np.max(preds, axis=1), -32)[-32:] if np.max(preds, axis=1).min() != 1 else None
    # Random baseline
    # print(preds.shape)
    # preds_arg = np.random.random_sample(preds_arg.shape)
    # preds_arg = preds_arg * (preds.shape[1])
    # print(preds_arg)
    # preds_arg = preds_arg.astype(int)
    # print(preds_arg)

    cost = metrics.confusion_matrix(preds_arg, labels)
    _, assignments = optimize.linear_sum_assignment(cost, maximize=True)
    for i in range(len(preds_arg)):
        preds_arg[i] = assignments[preds_arg[i]]
    match = labels == preds_arg

    total = 0
    n_clusters = preds_arg.max() + 1
    for i in range(n_clusters):
        true_mask = labels == i
        total += preds_arg[true_mask & match].shape[0] / np.sum(true_mask) / n_clusters

    print(f"Accuracy (normalized): {total}")
    print(f"Accuracy (raw): {np.sum(match) / match.shape[0]}")

    # plot_correctness(embeddings, labels, preds_arg, match)

def f(a,N):
    mask = np.empty(a.size,bool)
    mask[:N] = True
    np.not_equal(a[N:],a[:-N],out=mask[N:])
    return mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embedding_type", type=str, required=True, help="embedding type, one of the names in embeddings/")
    parser.add_argument("-f", "--filename", type=str, required=True, help="filename for cluster predictions")
    args = parser.parse_args()
    # embeddings, preds, labels = output["embeddings"], output["preds"], output["labels"]

    embeddings, labels, _ = load_embeddings(args.embedding_type)
    # bengals only full model
    # labels_arg = np.argmax(labels, axis=1)
    # mask = (labels_arg == 4) | (labels_arg == 5) | (labels_arg == 7) | (labels_arg == 8) | (labels_arg == 9) | (labels_arg == 13) | (labels_arg == 14)
    # embeddings = embeddings[mask]
    # labels = np.delete(labels[mask], [0, 1, 2, 3, 6, 10, 11, 12, 15], axis=1)

    # Test class imbalance
    # mask = f(np.argmax(labels, axis=1), 128)
    # embeddings = embeddings[mask]
    # labels = labels[mask]
    preds = load_predictions(args.filename)
    # bengals only full model
    # preds = np.delete(preds[mask], [0, 1, 2, 3, 6, 10, 11, 12, 15], axis=1)
    # preds, labels = np.load(f"{output_filename}/predictions.npy"), np.load(f"{output_filename}/labels.npy")
    evaluate_preds(embeddings, labels, preds)