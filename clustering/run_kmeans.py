import numpy as np
from sklearn import metrics, cluster
import hydra

from utils import load_embeddings, normalize_embeddings, to_onehot, save_output

from hydra.utils import get_original_cwd, to_absolute_path

@hydra.main(config_path="../config", config_name="config_baseline", version_base=None)
def run(cfg):
    n_clusters = cfg.n_clusters

    filename = get_original_cwd() + "/../embeddings/cls_tokens.npz"
    embeddings, labels = load_embeddings(filename)
    kmeans = cluster.KMeans(n_clusters=n_clusters).fit(normalize_embeddings(embeddings))

    save_output(embeddings, to_onehot(kmeans.labels_), labels)

if __name__ == "__main__":
    run()
