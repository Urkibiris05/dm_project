"""Barrido escalable de DBSCAN sobre representaciones por mensaje.

El barrido no construye una matriz completa de distancias: calcula vecinos
por fuerza bruta y bloques mediante scikit-learn. Las etiquetas externas solo
se guardan para la evaluación posterior.
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
VECTOR_DIR = ROOT / "vectorizations2"
RESULT_DIR = Path(__file__).resolve().parent / "resultados"
MIN_SAMPLES = (5, 10, 20, 40)
PERCENTILES = (10, 20, 30, 40, 50, 70)
SAMPLE_SEED = 20261021


def run_representation(
    name: str, vector_dir: Path, sample_size: int | None = None
) -> pd.DataFrame:
    """Ejecuta el barrido de una representación por mensaje."""
    folder = vector_dir / name
    X = np.load(folder / "X.npy", mmap_mode="r")
    metadata = pd.read_csv(folder / "metadata.csv")
    if sample_size is not None and sample_size < len(X):
        indices = np.sort(np.random.default_rng(SAMPLE_SEED).choice(
            len(X), size=sample_size, replace=False
        ))
        X = X[indices]
        metadata = metadata.iloc[indices].reset_index(drop=True)
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
                    "representation": name,
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
                f"{name} min_samples={min_samples} P{percentile}: "
                f"clusters={n_clusters} noise={noise_ratio:.1%}"
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation",
        nargs="+",
        help="Nombres de carpetas bajo vectorizations2.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ejecuta todas las carpetas message__all__ con X.npy.",
    )
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=VECTOR_DIR,
        help="Directorio que contiene las vectorizaciones.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Submuestra reproducible por representación para el cribado.",
    )
    args = parser.parse_args()
    vector_dir = args.vector_dir.resolve()
    if args.all:
        representations = sorted(
            path.name
            for path in vector_dir.glob("message__all__*")
            if (path / "X.npy").exists() and (path / "metadata.csv").exists()
        )
    elif args.representation:
        representations = args.representation
    else:
        parser.error("Indica --representation o --all")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.concat(
        [
            run_representation(name, vector_dir, args.sample_size)
            for name in representations
        ],
        ignore_index=True,
    )
    suffix = "_sample" if args.sample_size is not None else ""
    output = RESULT_DIR / f"e2_e3_message_sweep_vectorizations2{suffix}.csv"
    results.to_csv(output, index=False)
    print(f"Resultados guardados en {output}")


if __name__ == "__main__":
    main()
