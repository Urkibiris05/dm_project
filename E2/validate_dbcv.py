"""Valida la métrica DBCV usada por E2 sobre datos sintéticos."""

from pathlib import Path
import runpy

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = Path(__file__).resolve().parent / "resultados"
DBCV = runpy.run_path(str(Path(__file__).resolve().parent / "run_e2_user.py"))[
    "dbcv_score"
]


def main() -> None:
    cases = {
        "blobs": (
            make_blobs(
                n_samples=300, centers=3, cluster_std=0.45, random_state=0
            ),
            (0.25, 0.35, 0.45, 0.60),
        ),
        "moons": (
            make_moons(n_samples=300, noise=0.06, random_state=0),
            (0.08, 0.12, 0.16, 0.22),
        ),
        "circles": (
            make_circles(
                n_samples=300, factor=0.45, noise=0.04, random_state=0
            ),
            (0.08, 0.12, 0.18, 0.25),
        ),
        "noise": (
            (np.random.default_rng(0).normal(size=(300, 2)), np.zeros(300)),
            (0.02, 0.05, 0.10),
        ),
    }
    rows = []
    for name, ((X, target), eps_values) in cases.items():
        X = StandardScaler().fit_transform(X)
        distances = pairwise_distances(X, metric="euclidean")
        for eps in eps_values:
            labels = DBSCAN(
                eps=eps, min_samples=8, metric="euclidean"
            ).fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            rows.append(
                {
                    "dataset": name,
                    "eps": eps,
                    "n_clusters": n_clusters,
                    "noise_ratio": float(np.mean(labels == -1)),
                    "dbcv": DBCV(distances, labels, 8),
                    "ari": adjusted_rand_score(target, labels),
                }
            )
    output = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(RESULT_DIR / "dbcv_validation.csv", index=False)
    print(output.to_string(index=False))
    print(f"Resultados guardados en {RESULT_DIR / 'dbcv_validation.csv'}")


if __name__ == "__main__":
    main()
