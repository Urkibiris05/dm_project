"""Implementaciones de DBSCAN que se comparan.

Para añadir una nueva (por ejemplo nuestra búsqueda por bloques) basta con
escribir una función ``run_...`` y añadirla a ``IMPLEMENTATIONS``.

Todas las funciones reciben lo mismo y devuelven las etiquetas, con el
ruido marcado como -1.
"""

from collections.abc import Callable
from dataclasses import dataclass

import dbscan as dbscan_python
import mlpack
import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN

from dbscan_implementation import DBSCAN

Runner = Callable[[np.ndarray, float, int, str], np.ndarray]


@dataclass
class Implementation:
    """Una implementación de DBSCAN que entra en la comparación."""

    name: str                 # identificador que se guarda en el CSV
    label: str                # nombre que aparece en las gráficas
    run: Runner
    metrics: tuple[str, ...]  # distancias que admite
    max_features: int | None = None  # máximo de dimensiones (None = sin tope)


def run_ours_brute(
    X: np.ndarray, eps: float, min_samples: int, metric: str
) -> np.ndarray:
    model = DBSCAN(
        eps=eps, min_samples=min_samples, metric=metric, algorithm="brute"
    )
    return model.fit_predict(X)


def run_sklearn(
    X: np.ndarray, eps: float, min_samples: int, metric: str, algorithm: str
) -> np.ndarray:
    model = SklearnDBSCAN(
        eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm
    )
    return model.fit_predict(X)


def run_sklearn_brute(
    X: np.ndarray, eps: float, min_samples: int, metric: str
) -> np.ndarray:
    return run_sklearn(X, eps, min_samples, metric, "brute")


def run_sklearn_kd_tree(
    X: np.ndarray, eps: float, min_samples: int, metric: str
) -> np.ndarray:
    return run_sklearn(X, eps, min_samples, metric, "kd_tree")


def run_sklearn_ball_tree(
    X: np.ndarray, eps: float, min_samples: int, metric: str
) -> np.ndarray:
    return run_sklearn(X, eps, min_samples, metric, "ball_tree")


def run_dbscan_python(
    X: np.ndarray, eps: float, min_samples: int, metric: str
) -> np.ndarray:
    # Paquete "dbscan" de PyPI: implementación paralela, solo euclídea
    # y con 20 dimensiones como máximo.
    labels, _ = dbscan_python.DBSCAN(X, eps=eps, min_samples=min_samples)
    return np.asarray(labels)


def run_mlpack(
    X: np.ndarray, eps: float, min_samples: int, naive: bool
) -> np.ndarray:
    # mlpack solo trabaja con distancia euclídea.
    output = mlpack.dbscan(
        input_=X, epsilon=eps, min_size=min_samples, naive=naive
    )
    labels = np.asarray(output["assignments"]).astype(np.int64).ravel()
    # mlpack marca el ruido con el mayor entero posible.
    labels[(labels < 0) | (labels >= len(X))] = -1
    return labels


def run_mlpack_kd_tree(
    X: np.ndarray, eps: float, min_samples: int, metric: str
) -> np.ndarray:
    return run_mlpack(X, eps, min_samples, naive=False)


def run_mlpack_brute(
    X: np.ndarray, eps: float, min_samples: int, metric: str
) -> np.ndarray:
    return run_mlpack(X, eps, min_samples, naive=True)


# El orden fija el color de cada implementación en las gráficas: las nuevas
# se añaden al final para que las anteriores no cambien de color.
IMPLEMENTATIONS: list[Implementation] = [
    Implementation(
        "ours_brute", "Nuestro (fuerza bruta)", run_ours_brute,
        ("euclidean", "cosine"),
    ),
    Implementation(
        "sklearn_brute", "scikit-learn (brute)", run_sklearn_brute,
        ("euclidean", "cosine"),
    ),
    Implementation(
        "sklearn_kd_tree", "scikit-learn (kd_tree)", run_sklearn_kd_tree,
        ("euclidean",),
    ),
    Implementation(
        "sklearn_ball_tree", "scikit-learn (ball_tree)",
        run_sklearn_ball_tree, ("euclidean",),
    ),
    Implementation(
        "dbscan_python", "dbscan (PyPI, paralelo)", run_dbscan_python,
        ("euclidean",), max_features=20,
    ),
    Implementation(
        "mlpack_kd_tree", "mlpack (kd-tree)", run_mlpack_kd_tree,
        ("euclidean",),
    ),
    Implementation(
        "mlpack_brute", "mlpack (fuerza bruta)", run_mlpack_brute,
        ("euclidean",),
    ),
]


def get_implementation(name: str) -> Implementation:
    for implementation in IMPLEMENTATIONS:
        if implementation.name == name:
            return implementation
    raise ValueError(f"Implementación desconocida: {name!r}")
