"""Valida candidatas de mensajes con DBCV aproximado y estabilidad.

Las distancias se calculan sobre la misma muestra reproducible usada en el
cribado, evitando una matriz cuadrática sobre los 26.914 mensajes.
"""

from pathlib import Path
import runpy

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, pairwise_distances

dbcv_score = runpy.run_path(
    str(Path(__file__).resolve().parent / "run_e2_user.py")
)["dbcv_score"]


ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = ROOT / "vectorizations2"
RESULT_DIR = Path(__file__).resolve().parent / "resultados"
INPUT = RESULT_DIR / "e2_e3_message_sweep_vectorizations2_sample.csv"
SAMPLE_SIZE = 3000
SEED = 20261021
REPEATS = 10


def stable_score(
    X: np.ndarray, reference: np.ndarray, eps: float, min_samples: int
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    scores = []
    for _ in range(REPEATS):
        indices = rng.integers(0, len(X), len(X))
        labels = DBSCAN(
            eps=eps * rng.uniform(0.95, 1.05),
            min_samples=min_samples,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ).fit_predict(X[indices])
        scores.append(adjusted_rand_score(reference[indices], labels))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    sweep = pd.read_csv(INPUT)
    candidates = sweep[
        (sweep["n_clusters"] >= 2)
        & (sweep["noise_ratio"] <= 0.40)
        & (sweep["largest_cluster_frac"] < 0.70)
    ].copy()
    rows = []
    for _, candidate in candidates.iterrows():
        X = np.load(
            VECTOR_DIR / candidate["representation"] / "X.npy",
            mmap_mode="r",
        )
        indices = np.sort(np.random.default_rng(SEED).choice(
            len(X), size=SAMPLE_SIZE, replace=False
        ))
        X = np.asarray(X[indices])
        distances = pairwise_distances(X, metric="cosine")
        labels = DBSCAN(
            eps=float(candidate["eps"]),
            min_samples=int(candidate["min_samples"]),
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ).fit_predict(X)
        dbcv = dbcv_score(
            distances, labels, int(candidate["min_samples"])
        )
        mean, std = stable_score(
            X,
            labels,
            float(candidate["eps"]),
            int(candidate["min_samples"]),
        )
        rows.append({
            **candidate.to_dict(),
            "dbcv_sample": dbcv,
            "stability_mean": mean,
            "stability_std": std,
        })
        print(
            f"{candidate['representation']} "
            f"min_samples={int(candidate['min_samples'])} "
            f"P{int(candidate['percentile'])}: DBCV={dbcv:.4f}, "
            f"stability={mean:.3f}",
            flush=True,
        )

    output = RESULT_DIR / "e2_e3_message_candidates_validation.csv"
    pd.DataFrame(rows).sort_values(
        ["dbcv_sample", "stability_mean"], ascending=False
    ).to_csv(output, index=False)
    print(f"Resultados guardados en {output}")


if __name__ == "__main__":
    main()
