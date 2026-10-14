"""Búsqueda de vecinos por fuerza bruta, calculada por bloques."""

from typing import Self

import numpy as np

from ..distances import Distance
from .base import NeighborSearch

# Memoria que ocupa como máximo un bloque de distancias (en bytes).
BLOCK_BYTES: int = 256 * 2**20


class BlockedBruteForceSearch(NeighborSearch):
    """Calcula la distancia de cada punto contra todos, pero por bloques.

    Se comparan ``block_size`` puntos contra todos los demás, se guardan solo
    los vecinos que caen dentro de ``eps`` y se descarta el bloque. Así nunca
    existe la matriz n x n completa.

    Coste, con n puntos, d dimensiones, bloques de b filas y k vecinos de
    media por punto:
      - tiempo:  O(n^2 * d), igual que la fuerza bruta completa;
      - memoria: O(b * n + n * k), en vez de O(n^2).

    Parámetros
    ----------
    block_size : int, opcional
        Filas por bloque. Por defecto se elige para que un bloque ocupe unos
        256 MB.
    """

    def __init__(self, block_size: int | None = None) -> None:
        if block_size is not None and block_size < 1:
            raise ValueError(
                f"block_size debe ser al menos 1 (block_size={block_size})"
            )
        self.block_size = block_size

    def fit(self, X: np.ndarray, distance: Distance) -> Self:
        self.X_: np.ndarray = X
        self.distance_: Distance = distance
        if self.block_size is None:
            # Un bloque de b filas ocupa b * n valores de 8 bytes.
            rows = BLOCK_BYTES // (8 * max(len(X), 1))
            self.block_size_: int = max(1, rows)
        else:
            self.block_size_ = self.block_size
        return self

    def radius_neighbors(self, eps: float) -> list[np.ndarray]:
        neighborhoods: list[np.ndarray] = []
        n_points = len(self.X_)
        for start in range(0, n_points, self.block_size_):
            block = self.X_[start:start + self.block_size_]
            distances = self.distance_.pairwise(block, self.X_)
            for row in distances:
                neighborhoods.append(np.flatnonzero(row <= eps))
        return neighborhoods
