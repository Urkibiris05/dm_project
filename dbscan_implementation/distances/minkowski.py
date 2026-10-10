"""Distancia de Minkowski (L_p)."""

import numpy as np
from numpy.typing import ArrayLike

from .base import Distance


class MinkowskiDistance(Distance):
    """Distancia L_p: ``d(a, b) = (sum_k |a_k - b_k|^p)^(1/p)``.

    Casos particulares:
      - ``p = 1``      -> Manhattan
      - ``p = 2``      -> euclídea
      - ``p = np.inf`` -> Chebyshev (máxima diferencia en una coordenada)

    También se admiten valores fraccionarios (por ejemplo ``p = 0.5``).
    Con ``p < 1`` ya no es una distancia en sentido estricto (no cumple la
    desigualdad triangular), pero DBSCAN solo necesita comparar con ``eps``.
    """

    def __init__(self, p: float = 2) -> None:
        if p <= 0:
            raise ValueError(f"p debe ser mayor que 0, pero se recibió p={p}")
        self.p = p

    def pairwise(self, points_a: ArrayLike, points_b: ArrayLike) -> np.ndarray:
        points_a = np.asarray(points_a, dtype=float)
        points_b = np.asarray(points_b, dtype=float)

        distances = np.empty((len(points_a), len(points_b)))
        # Se recorre points_a fila a fila: restar todo de golpe crearía un
        # array de tamaño n_a x n_b x n_dimensiones, que no cabe en memoria.
        for i, point in enumerate(points_a):
            differences = np.abs(points_b - point)  # (n_b, n_dimensiones)
            if np.isinf(self.p):
                distances[i] = differences.max(axis=1)
            else:
                sums = (differences ** self.p).sum(axis=1)
                distances[i] = sums ** (1 / self.p)
        return distances

    def __repr__(self) -> str:
        return f"MinkowskiDistance(p={self.p})"
