"""E4: compara métricas de distancia sobre todos los mensajes."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "vectorizations2" / "message__all__e5"
RESULTS = Path(__file__).resolve().parent / "resultados"
MIN_SAMPLES = 5
PERCENTILE = 40


def external_scores(metadata: pd.DataFrame, labels: np.ndarray) -> dict:
    return {
        "label_ari": adjusted_rand_score(
            metadata["label"].astype("category").cat.codes, labels
        ),
        "label_nmi": normalized_mutual_info_score(
            metadata["label"].astype("category").cat.codes, labels
        ),
    }


def main() -> None:
    X = np.load(FOLDER / "X.npy", mmap_mode="r")
    metadata = pd.read_csv(FOLDER / "metadata.csv")
    rows = []
    for metric, kwargs in (
        ("cosine", {}),
        ("minkowski", {"p": 1}),
        ("minkowski", {"p": 2}),
        ("minkowski", {"p": np.inf}),
    ):
        neighbors = NearestNeighbors(
            n_neighbors=MIN_SAMPLES,
            metric=metric,
            metric_params=kwargs or None,
            algorithm="brute",
            n_jobs=-1,
        ).fit(X)
        eps = float(np.percentile(
            neighbors.kneighbors(X, return_distance=True)[0][:, -1],
            PERCENTILE,
        ))
        labels = DBSCAN(
            eps=eps,
            min_samples=MIN_SAMPLES,
            metric=metric,
            metric_params=kwargs or None,
            algorithm="brute",
            n_jobs=-1,
        ).fit_predict(X)
        sizes = np.bincount(labels[labels >= 0])
        rows.append({
            "experiment": "E4",
            "metric": metric if not kwargs else f"minkowski_p_{kwargs['p']}",
            "eps": eps,
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "noise_ratio": float(np.mean(labels == -1)),
            "largest_cluster_frac": (
                float(sizes.max() / len(labels)) if len(sizes) else 0.0
            ),
            **external_scores(metadata, labels),
        })
    output = RESULTS / "e4_message_distances.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
