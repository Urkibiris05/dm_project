"""Tests del módulo E4."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from E4.run_e4 import run_experiment, save_results


def build_vectorization(path: Path) -> Path:
    """Crea una vectorización mínima válida para las pruebas."""
    X, y = make_blobs(
        n_samples=60,
        centers=3,
        n_features=5,
        cluster_std=0.45,
        random_state=0,
    )
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "X.npy", X)
    pd.DataFrame(
        {
            "label": np.where(y == 0, "alto", "bajo"),
            "disorder": np.where(y == 2, "Depresion", "Ansiedad"),
        }
    ).to_csv(path / "metadata.csv", index=False)
    return path


def test_run_experiment_and_save_results(tmp_path: Path) -> None:
    """E4 genera resultados y los guarda en CSV."""
    vectorization = build_vectorization(tmp_path / "vec")
    results = run_experiment(
        vectorization=vectorization,
        metric="euclidean",
        min_samples_values=(5, 10),
        percentiles=(20, 50),
    )

    assert len(results) == 4
    assert {"dbcv", "noise_rate", "label_ari_all"}.issubset(results.columns)

    grid_path, best_path = save_results(results, tmp_path / "out")
    assert grid_path.exists()
    assert best_path.exists()
    assert len(pd.read_csv(best_path)) == 1
