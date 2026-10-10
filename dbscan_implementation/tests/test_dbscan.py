"""Tests de DBSCAN. scikit-learn solo se usa aquí, como comprobación."""

import numpy as np
import pytest
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.datasets import make_blobs, make_moons

from dbscan_implementation import DBSCAN
from dbscan_implementation.distances import MinkowskiDistance
from dbscan_implementation.neighbors import BruteForceSearch


def make_datasets() -> dict[str, np.ndarray]:
    blobs, _ = make_blobs(
        n_samples=300, centers=4, cluster_std=0.8, random_state=0
    )
    moons, _ = make_moons(n_samples=300, noise=0.08, random_state=0)
    return {"blobs": blobs, "moons": moons}


DATASETS = make_datasets()


@pytest.mark.parametrize("dataset", ["blobs", "moons"])
@pytest.mark.parametrize(
    "params",
    [
        {"eps": 0.5, "min_samples": 5, "metric": "euclidean"},
        {"eps": 0.2, "min_samples": 5, "metric": "euclidean"},
        {"eps": 0.3, "min_samples": 10, "metric": "euclidean"},
        {"eps": 0.5, "min_samples": 5, "metric": "manhattan"},
        {"eps": 0.3, "min_samples": 4, "metric": "chebyshev"},
        {"eps": 0.4, "min_samples": 5, "metric": "minkowski", "p": 3},
        {"eps": 0.01, "min_samples": 5, "metric": "cosine"},
    ],
)
def test_matches_sklearn(dataset: str, params: dict) -> None:
    X = DATASETS[dataset]

    ours = DBSCAN(**params).fit(X)
    theirs = SklearnDBSCAN(algorithm="brute", **params).fit(X)

    assert np.array_equal(ours.labels_, theirs.labels_)
    assert np.array_equal(
        ours.core_sample_indices_, theirs.core_sample_indices_
    )
    assert np.array_equal(ours.components_, theirs.components_)


def test_cosine_equals_euclidean_on_normalized_vectors() -> None:
    # Si |x| = |y| = 1, entonces |x - y|^2 = 2 * (1 - cos(x, y)).
    X = np.random.default_rng(0).normal(size=(200, 3))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    eps_cosine = 0.05
    labels_cosine = DBSCAN(
        eps=eps_cosine, min_samples=4, metric="cosine"
    ).fit_predict(X)
    labels_euclidean = DBSCAN(
        eps=np.sqrt(2 * eps_cosine), min_samples=4, metric="euclidean"
    ).fit_predict(X)

    assert len(set(labels_cosine)) > 1  # el caso no es trivial
    assert np.array_equal(labels_cosine, labels_euclidean)


def test_small_example() -> None:
    X = [[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]
    labels = DBSCAN(eps=3, min_samples=2).fit_predict(X)
    assert labels.tolist() == [0, 0, 0, 1, 1, -1]


def test_all_noise() -> None:
    X = DATASETS["blobs"]
    model = DBSCAN(eps=1e-6, min_samples=5).fit(X)
    assert np.all(model.labels_ == -1)
    assert len(model.core_sample_indices_) == 0
    assert model.components_.shape == (0, X.shape[1])


def test_single_cluster() -> None:
    X = DATASETS["blobs"]
    labels = DBSCAN(eps=100, min_samples=5).fit_predict(X)
    assert np.all(labels == 0)


def test_min_samples_one_has_no_noise() -> None:
    X = DATASETS["blobs"]
    model = DBSCAN(eps=0.1, min_samples=1).fit(X)
    assert np.all(model.labels_ >= 0)
    assert len(model.core_sample_indices_) == len(X)


def test_accepts_distance_and_search_objects() -> None:
    X = DATASETS["moons"]
    by_name = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)
    by_object = DBSCAN(
        eps=0.2,
        min_samples=5,
        metric=MinkowskiDistance(p=2),
        algorithm=BruteForceSearch(),
    ).fit_predict(X)
    assert np.array_equal(by_name, by_object)


@pytest.mark.parametrize(
    "params",
    [
        {"metric": "no_existe"},
        {"algorithm": "no_existe"},
        {"eps": 0},
        {"min_samples": 0},
    ],
)
def test_invalid_parameters(params: dict) -> None:
    with pytest.raises(ValueError):
        DBSCAN(**params).fit(DATASETS["blobs"])
