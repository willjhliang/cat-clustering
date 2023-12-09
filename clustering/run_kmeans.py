import numpy as np
from sklearn import metrics, cluster
import hydra

from utils import load_embeddings, normalize_embeddings, to_onehot, save_output
from evaluate_preds import evaluate_preds

from hydra.utils import get_original_cwd, to_absolute_path

@hydra.main(config_path="../config", config_name="config_baseline", version_base=None)
def run(cfg):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    n_clusters = cfg.n_clusters
    embedding_type = cfg.embedding_type

    embeddings, labels, _ = load_embeddings(embedding_type)
    kmeans = cluster.KMeans(n_clusters=n_clusters, n_init="auto").fit(normalize_embeddings(embeddings))

    # save_output(embeddings, to_onehot(kmeans.labels_), labels)
    save_output(to_onehot(kmeans.labels_), save_dir=output_dir)

    evaluate_preds(embeddings, labels, to_onehot(kmeans.labels_))

if __name__ == "__main__":
    run()
