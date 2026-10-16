"""Datos de juguete de scikit-learn para E0 (no se usa el corpus real)."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import (
    load_digits,
    load_iris,
    load_wine,
    make_blobs,
    make_circles,
    make_moons,
)
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from .config import EPS_QUANTILES, N_SAMPLES, RANDOM_STATE


@dataclass
class Dataset:
    """Un conjunto de datos. ``y`` solo sirve para colorear figuras."""

    name: str
    X: np.ndarray
    y: np.ndarray


def _scaled(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def normalize_rows(X: np.ndarray) -> np.ndarray:
    """Divide cada fila entre su norma (las filas nulas no cambian)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def load_datasets() -> dict[str, Dataset]:
    """Devuelve todos los datasets del experimento, por nombre."""
    rs = RANDOM_STATE
    datasets: dict[str, Dataset] = {}

    X, y = make_blobs(N_SAMPLES, centers=3, random_state=rs)
    datasets["blobs"] = Dataset("blobs", _scaled(X), y)

    X, y = make_blobs(
        N_SAMPLES, cluster_std=[1.0, 2.5, 0.5], random_state=rs
    )
    datasets["varied"] = Dataset("varied", _scaled(X), y)

    X, y = make_blobs(N_SAMPLES, centers=3, random_state=rs)
    X = X @ np.array([[0.6, -0.6], [-0.4, 0.8]])
    datasets["aniso"] = Dataset("aniso", _scaled(X), y)

    X, y = make_moons(N_SAMPLES, noise=0.05, random_state=rs)
    datasets["moons"] = Dataset("moons", _scaled(X), y)

    X, y = make_circles(N_SAMPLES, factor=0.5, noise=0.05, random_state=rs)
    datasets["circles"] = Dataset("circles", _scaled(X), y)

    X = np.random.default_rng(rs).random((N_SAMPLES, 2))
    datasets["no_structure"] = Dataset(
        "no_structure", _scaled(X), np.zeros(N_SAMPLES, dtype=int)
    )

    iris = load_iris()
    datasets["iris"] = Dataset("iris", _scaled(iris.data), iris.target)
    wine = load_wine()
    datasets["wine"] = Dataset("wine", _scaled(wine.data), wine.target)
    digits = load_digits()
    datasets["digits"] = Dataset(
        "digits", _scaled(digits.data), digits.target
    )

    # Simula embeddings de frases: 384 dimensiones y norma 1.
    X, y = make_blobs(
        2000, n_features=384, centers=5, cluster_std=1.0,
        center_box=(-1, 1), random_state=rs,
    )
    datasets["emb_384d"] = Dataset("emb_384d", normalize_rows(X), y)
    return datasets


def distance_matrix(
    X: np.ndarray, metric: str, p: float | None = None
) -> np.ndarray:
    """Matriz de distancias completa calculada con scikit-learn."""
    if metric == "minkowski":
        return pairwise_distances(X, metric=metric, p=p)
    return pairwise_distances(X, metric=metric)


def kdist_from_matrix(D: np.ndarray, min_samples: int) -> np.ndarray:
    """Distancia al min_samples-ésimo vecino (el propio punto cuenta).

    La columna 0 de la matriz ordenada es el propio punto, igual que
    min_samples cuenta al punto en DBSCAN.
    """
    return np.sort(D, axis=1)[:, min_samples - 1]


def eps_grid(
    D: np.ndarray,
    min_samples: int,
    quantiles: np.ndarray | tuple[float, ...] = EPS_QUANTILES,
) -> np.ndarray:
    """Valores de eps = cuantiles del k-dist.

    Con interpolación lineal un cuantil puede caer justo sobre una distancia
    real, lo que provoca empates en la frontera a propósito (prueba de
    robustez numérica).
    """
    return np.quantile(kdist_from_matrix(D, min_samples), quantiles)
