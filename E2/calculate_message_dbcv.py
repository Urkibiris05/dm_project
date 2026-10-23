"""Calcula DBCV para las configuraciones válidas del barrido de mensajes.

Las configuraciones se seleccionan con estadísticas del corpus completo. El
DBCV se evalúa sobre una muestra fija por representación porque su cálculo
requiere una matriz de distancias cuadrática.
"""

from pathlib import Path
import runpy

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances


ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = ROOT / "vectorizations2"
RESULT_DIR = Path(__file__).resolve().parent / "resultados"
SWEEP = RESULT_DIR / "e2_message_hierarchical_sweep.csv"
SAMPLE_SIZE = 3000
SEED = 20261021

dbcv_score = runpy.run_path(
    str(Path(__file__).resolve().parent / "run_e2_user.py")
)["dbcv_score"]


def main() -> None:
    sweep = pd.read_csv(SWEEP)
    valid = sweep[
        (sweep["n_message_clusters"] >= 2)
        & (sweep["message_noise_ratio"] <= 0.40)
        & (sweep["largest_message_cluster_frac"] < 0.70)
    ].copy()
    rows: list[dict[str, object]] = []

    for representation, candidates in valid.groupby("representation", sort=True):
        X = np.load(
            VECTOR_DIR / representation / "X.npy",
            mmap_mode="r",
        )
        rng = np.random.default_rng(SEED)
        indices = np.sort(rng.choice(len(X), size=min(SAMPLE_SIZE, len(X)), replace=False))
        sample = np.asarray(X[indices])
        distances = pairwise_distances(sample, metric="cosine")
        for _, candidate in candidates.iterrows():
            labels = DBSCAN(
                eps=float(candidate["eps_message"]),
                min_samples=int(candidate["min_samples_message"]),
                metric="cosine",
                algorithm="brute",
                n_jobs=-1,
            ).fit_predict(sample)
            clusters = len(set(labels)) - int(-1 in labels)
            row = candidate.to_dict()
            row["dbcv_sample"] = dbcv_score(
                distances, labels, int(candidate["min_samples_message"])
            )
            row["sample_n_clusters"] = clusters
            row["sample_noise_ratio"] = float(np.mean(labels == -1))
            rows.append(row)
        print(f"{representation}: {len(candidates)} configuraciones", flush=True)

    output = RESULT_DIR / "e2_message_valid_dbcv.csv"
    pd.DataFrame(rows).sort_values(
        ["dbcv_sample", "representation"],
        ascending=[False, True],
    ).to_csv(output, index=False)
    print(f"Resultados guardados en {output}")


if __name__ == "__main__":
    main()
