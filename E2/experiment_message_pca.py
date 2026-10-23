"""Estudia el efecto de reducir E5 antes de aplicar DBSCAN."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "vectorizations2" / "message__all__e5"
OUTPUT = ROOT / "E2" / "resultados" / "e2_message_e5_pca_sample.csv"
SAMPLE_SIZE = 3000
SEED = 20261021
MIN_SAMPLES = (5, 10, 20, 40)
PERCENTILES = (10, 20, 30, 40, 50, 70)
DIMS = (32, 64, 128, 384)


def normalize(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def main() -> None:
    raw = np.load(FOLDER / "X.npy", mmap_mode="r")
    metadata = pd.read_csv(FOLDER / "metadata.csv")
    indices = np.sort(np.random.default_rng(SEED).choice(
        len(raw), size=SAMPLE_SIZE, replace=False
    ))
    raw = np.asarray(raw[indices])
    metadata = metadata.iloc[indices]
    label = metadata["label"].astype("category").cat.codes.to_numpy()
    disorder = metadata["disorder"].astype("category").cat.codes.to_numpy()
    rows = []
    for dim in DIMS:
        X = raw if dim == 384 else PCA(
            n_components=dim, random_state=SEED
        ).fit_transform(raw)
        X = normalize(X)
        for min_samples in MIN_SAMPLES:
            distances, _ = NearestNeighbors(
                n_neighbors=min_samples, metric="cosine",
                algorithm="brute", n_jobs=-1,
            ).fit(X).kneighbors(X)
            kdist = distances[:, -1]
            for percentile in PERCENTILES:
                eps = float(np.percentile(kdist, percentile))
                clusters = DBSCAN(
                    eps=eps, min_samples=min_samples, metric="cosine",
                    algorithm="brute", n_jobs=-1,
                ).fit_predict(X)
                sizes = np.bincount(clusters[clusters >= 0])
                rows.append({
                    "dimension": dim,
                    "min_samples": min_samples,
                    "percentile": percentile,
                    "eps": eps,
                    "n_clusters": len(set(clusters)) - (
                        1 if -1 in clusters else 0
                    ),
                    "noise_ratio": float(np.mean(clusters == -1)),
                    "largest_cluster_frac": (
                        float(sizes.max() / len(X)) if len(sizes) else 0.0
                    ),
                    "label_ari": adjusted_rand_score(label, clusters),
                    "label_nmi": normalized_mutual_info_score(
                        label, clusters
                    ),
                    "disorder_ari": adjusted_rand_score(disorder, clusters),
                    "disorder_nmi": normalized_mutual_info_score(
                        disorder, clusters
                    ),
                })
        print(f"dimension={dim}", flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"Resultados guardados en {OUTPUT}")


if __name__ == "__main__":
    main()
