"""Piloto escalable de DBSCAN sobre embeddings por mensaje.

Se ejecuta un encoder cada vez para no materializar matrices cuadráticas.
Ejemplo:

    python E2/run_e2_messages.py --encoder minilm
"""

import argparse
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
VECTOR_DIR = ROOT / "vectorizations"
RESULT_DIR = Path(__file__).resolve().parent / "resultados"
MIN_SAMPLES = (5, 10, 20, 40)
PERCENTILES = (10, 20, 30, 40, 50, 70)


def run_encoder(encoder: str) -> pd.DataFrame:
    """Ejecuta el barrido de un embedding por mensaje."""
    name = f"message__all__{encoder}"
    folder = VECTOR_DIR / name
    X = np.load(folder / "X.npy", mmap_mode="r")
    metadata = pd.read_csv(folder / "metadata.csv")
    label = metadata["label"].astype("category").cat.codes.to_numpy()
    disorder = metadata["disorder"].astype("category").cat.codes.to_numpy()
    rows = []

    for min_samples in MIN_SAMPLES:
        neighbor_distances, _ = NearestNeighbors(
            n_neighbors=min_samples,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ).fit(X).kneighbors(X)
        kdist = np.sort(neighbor_distances[:, -1])
        for percentile in PERCENTILES:
            eps = float(np.percentile(kdist, percentile))
            clusters = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric="cosine",
                algorithm="brute",
                n_jobs=-1,
            ).fit_predict(X)
            n_clusters = len(set(clusters)) - (
                1 if -1 in clusters else 0
            )
            noise_ratio = float(np.mean(clusters == -1))
            sizes = np.bincount(clusters[clusters >= 0])
            largest = float(sizes.max() / len(clusters)) if len(sizes) else 0.0
            rows.append(
                {
                    "encoder": encoder,
                    "min_samples": min_samples,
                    "percentile": percentile,
                    "eps": eps,
                    "n_clusters": n_clusters,
                    "noise_ratio": noise_ratio,
                    "largest_cluster_frac": largest,
                    "label_ari": adjusted_rand_score(label, clusters),
                    "label_nmi": normalized_mutual_info_score(
                        label, clusters
                    ),
                    "disorder_ari": adjusted_rand_score(disorder, clusters),
                    "disorder_nmi": normalized_mutual_info_score(
                        disorder, clusters
                    ),
                }
            )
            print(
                f"{encoder} min_samples={min_samples} P{percentile}: "
                f"clusters={n_clusters} noise={noise_ratio:.1%}"
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder", choices=("minilm", "e5", "beto-sts"), required=True
    )
    args = parser.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_encoder(args.encoder)
    output = RESULT_DIR / f"e2_message_{args.encoder}_pilot.csv"
    results.to_csv(output, index=False)
    print(f"Resultados guardados en {output}")


if __name__ == "__main__":
    main()
