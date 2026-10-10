"""Distancias disponibles para DBSCAN.

Para añadir una nueva:
  1. crear un archivo en esta carpeta con una clase que herede de ``Distance``;
  2. añadir su nombre a ``get_distance``.
"""

import numpy as np

from .base import Distance
from .cosine import CosineDistance
from .minkowski import MinkowskiDistance

__all__ = ["Distance", "MinkowskiDistance", "CosineDistance", "get_distance"]

# Nombres admitidos en el parámetro ``metric`` (como en scikit-learn).
METRIC_NAMES: list[str] = [
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "cosine",
]


def get_distance(
    metric: str | Distance = "euclidean",
    p: float | None = None,
) -> Distance:
    """Devuelve el objeto ``Distance`` que corresponde a ``metric``.

    Parámetros
    ----------
    metric : str o Distance
        Nombre de la distancia, o directamente un objeto ``Distance``.
    p : float, opcional
        Exponente de Minkowski. Solo se usa con ``metric="minkowski"``
        (por defecto 2, la euclídea).
    """
    if isinstance(metric, Distance):
        return metric

    if metric == "euclidean":
        return MinkowskiDistance(p=2)
    if metric == "manhattan":
        return MinkowskiDistance(p=1)
    if metric == "chebyshev":
        return MinkowskiDistance(p=np.inf)
    if metric == "minkowski":
        return MinkowskiDistance(p=2 if p is None else p)
    if metric == "cosine":
        return CosineDistance()

    raise ValueError(
        f"Distancia desconocida: {metric!r}. Opciones: {METRIC_NAMES}"
    )
