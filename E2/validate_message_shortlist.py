"""Valida en el corpus completo las mejores candidatas del cribado."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = ROOT / "vectorizations2"
RESULT_DIR = Path(__file__).resolve().parent / "resultados"
SHORTLIST = (
    "message__all__e5",
    "message__all__tfidf-char_wb-1-5-svd128",
    "message__all__robertalex",
    "message__all__beto-sts",
)
MIN_SAMPLES = {
    "message__all__e5": 5,
    "message__all__tfidf-char_wb-1-5-svd128": 5,
    "message__all__robertalex": 5,
    "message__all__beto-sts": 5,
}
PERCENTILE = 40
EPS_FACTORS = (0.95, 1.0, 1.05)


def main() -> None:
    rows = []
    for name in SHORTLIST:
        folder = VECTOR_DIR / name
        X = np.load(folder / "X.npy", mmap_mode="r")
        metadata = pd.read_csv(folder / "metadata.csv")
        label = metadata["label"].astype("category").cat.codes.to_numpy()
        disorder = metadata["disorder"].astype("category").cat.codes.to_numpy()
        min_samples = MIN_SAMPLES[name]
        neighbors = NearestNeighbors(
            n_neighbors=min_samples, metric="cosine",
            algorithm="brute", n_jobs=-1,
        ).fit(X)
        distances, _ = neighbors.kneighbors(X)
        eps = float(np.percentile(distances[:, -1], PERCENTILE))
        for factor in EPS_FACTORS:
            current_eps = eps * factor
            clusters = DBSCAN(
                eps=current_eps, min_samples=min_samples,
                metric="cosine", algorithm="brute", n_jobs=-1,
            ).fit_predict(X)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            sizes = np.bincount(clusters[clusters >= 0])
            rows.append({
                "representation": name,
                "n_points": len(X),
                "min_samples": min_samples,
                "percentile": PERCENTILE,
                "eps": current_eps,
                "eps_factor": factor,
                "n_clusters": n_clusters,
                "noise_ratio": float(np.mean(clusters == -1)),
                "largest_cluster_frac": (
                    float(sizes.max() / len(X)) if len(sizes) else 0.0
                ),
                "label_ari": adjusted_rand_score(label, clusters),
                "label_nmi": normalized_mutual_info_score(label, clusters),
                "disorder_ari": adjusted_rand_score(disorder, clusters),
                "disorder_nmi": normalized_mutual_info_score(
                    disorder, clusters
                ),
            })
            print(
                f"{name} factor={factor}: clusters={n_clusters}, "
                f"noise={rows[-1]['noise_ratio']:.1%}",
                flush=True,
            )
    output = RESULT_DIR / "e2_message_shortlist_full_validation.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Resultados guardados en {output}")


if __name__ == "__main__":
    main()
