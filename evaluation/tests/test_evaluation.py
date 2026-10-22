"""Tests del paquete evaluation."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

from evaluation.dbcv import dbcv
from evaluation.kdist import (eps_grid, kdist, knee_point, knee_sharpness,
                              relative_contrast)
from evaluation.metrics import (external_metrics, internal_metrics,
                                structure_summary)
from evaluation.selection import select, shortlist
from evaluation.stability import bootstrap_stability, eps_perturbation


def blobs(n_dims: int = 2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(
        n_samples=300, centers=3, n_features=n_dims, cluster_std=0.4,
        center_box=(-10, 10), random_state=seed)
    return X, y


def test_kdist_matches_sklearn() -> None:
    X = np.random.default_rng(0).normal(size=(200, 5))
    expected = NearestNeighbors(n_neighbors=6).fit(X).kneighbors(X)[0][:, 5]
    result = kdist(X, k=6, metric="euclidean", block_size=64)
    assert np.allclose(result, expected)


def test_knee_of_l_shaped_curve() -> None:
    curve = np.concatenate([np.linspace(0.1, 0.2, 90), np.linspace(1, 5, 10)])
    index, value = knee_point(curve)
    assert 85 <= index <= 95
    assert knee_sharpness(curve) > knee_sharpness(np.linspace(0, 1, 100))
    assert value > 0.15


def test_eps_grid_avoids_exact_quantile() -> None:
    kd = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.all(eps_grid(kd, [0.5]) > np.quantile(kd, 0.5))


def test_relative_contrast_is_positive() -> None:
    X = np.random.default_rng(0).normal(size=(100, 10))
    assert relative_contrast(X, "euclidean", n_queries=20) > 0


def test_dbcv_separated_blobs_is_high() -> None:
    X, y = blobs()
    assert dbcv(X, y, "euclidean") > 0.7


def test_dbcv_random_labels_is_low() -> None:
    X, y = blobs()
    shuffled = np.random.default_rng(0).permutation(y)
    assert dbcv(X, shuffled, "euclidean") < 0.0


def test_dbcv_noise_lowers_score() -> None:
    X, y = blobs()
    noisy = y.copy()
    noisy[:60] = -1
    assert dbcv(X, noisy, "euclidean") < dbcv(X, y, "euclidean")


def test_dbcv_high_dimension_is_finite() -> None:
    X, y = blobs(n_dims=300)
    assert np.isfinite(dbcv(X, y, "euclidean"))


def test_dbcv_single_cluster_is_nan() -> None:
    X, _ = blobs()
    assert np.isnan(dbcv(X, np.zeros(len(X), dtype=int), "euclidean"))


def test_structure_and_metrics() -> None:
    X, y = blobs()
    labels = y.copy()
    labels[:30] = -1
    summary = structure_summary(labels)
    assert summary["n_clusters"] == 3
    assert abs(summary["noise_rate"] - 0.1) < 1e-9
    internal = internal_metrics(X, labels, "euclidean")
    assert internal["silhouette"] > 0.5
    external = external_metrics(labels, y)
    assert external["ari_nonoise"] > 0.99


def test_selection_rejects_noisy_best() -> None:
    grid = pd.DataFrame({
        "space": ["a"] * 3,
        "min_samples": [5, 10, 20],
        "n_clusters": [3, 3, 2],
        "noise_rate": [0.6, 0.3, 0.2],
        "largest_cluster_frac": [0.3, 0.4, 0.5],
        "dbcv": [0.9, 0.5, 0.4],
        "stab_p10": [0.9, 0.9, 0.9],
        "eps_ari": [0.9, 0.9, 0.9],
    })
    chosen = select(grid, ["space"])
    assert chosen.loc[0, "min_samples"] == 10
    assert chosen.loc[0, "status"] == "ok"
    assert list(shortlist(grid, ["space"])) == [1, 2]


def test_selection_without_valid_is_no_structure() -> None:
    grid = pd.DataFrame({
        "space": ["a", "a"],
        "min_samples": [5, 10],
        "n_clusters": [1, 2],
        "noise_rate": [0.1, 0.5],
        "largest_cluster_frac": [0.9, 0.5],
        "dbcv": [float("nan"), -0.1],
    })
    chosen = select(grid, ["space"])
    assert chosen.loc[0, "status"] == "sin_estructura"


def test_selection_unstable_status() -> None:
    grid = pd.DataFrame({
        "space": ["a"],
        "min_samples": [5],
        "n_clusters": [3],
        "noise_rate": [0.1],
        "largest_cluster_frac": [0.4],
        "dbcv": [0.5],
        "stab_p10": [0.2],
        "eps_ari": [0.9],
    })
    assert select(grid, ["space"]).loc[0, "status"] == "inestable"


def test_stability_of_separated_blobs() -> None:
    X, _ = blobs()
    result = bootstrap_stability(
        X, eps=1.0, min_samples=5, metric="euclidean", n_runs=5)
    assert result["stab_mean"] > 0.95
    assert eps_perturbation(X, 1.0, 5, "euclidean") > 0.95
