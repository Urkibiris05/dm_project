"""Búsqueda de vecinos por fuerza bruta."""

from typing import Self

import numpy as np

from ..distances import Distance
from .base import NeighborSearch


class BruteForceSearch(NeighborSearch):
    """Calcula la distancia de cada punto contra todos los demás.

    Guarda la matriz completa de distancias (n x n), así que el coste es
    O(n^2) en tiempo y en memoria. Sirve para conjuntos pequeños o medianos
    (miles de puntos); para más hará falta una búsqueda por bloques que no
    guarde la matriz entera.
    """

    def fit(self, X: np.ndarray, distance: Distance) -> Self:
        self.distance_matrix_: np.ndarray = distance.pairwise(X, X)
        return self

    def radius_neighbors(self, eps: float) -> list[np.ndarray]:
        neighborhoods: list[np.ndarray] = []
        for row in self.distance_matrix_:
            neighborhoods.append(np.flatnonzero(row <= eps))
        return neighborhoods
