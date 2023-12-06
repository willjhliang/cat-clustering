import numpy as np
from sklearn import metrics
from scipy import optimize
import sys

from utils import plot_correctness, load_output

def evaluate_preds(embeddings, labels, preds):
    labels, preds_arg = np.argmax(labels, axis=1), np.argmax(preds, axis=1)
    top = np.argpartition(np.max(preds, axis=1), -32)[-32:] if np.max(preds, axis=1).min() != 1 else None

    cost = metrics.confusion_matrix(preds_arg, labels)
    _, assignments = optimize.linear_sum_assignment(cost, maximize=True)
    for i in range(len(preds_arg)):
        preds_arg[i] = assignments[preds_arg[i]]
    match = labels == preds_arg

    print(f"Accuracy: {np.sum(match) / match.shape[0]}")

    plot_correctness(embeddings, labels, preds_arg, match, top)

if __name__ == "__main__":
    output_filename = sys.argv[1]
    output = load_output(output_filename)
    embeddings, preds, labels = output["embeddings"], output["preds"], output["labels"]
    evaluate_preds(embeddings, labels, preds)