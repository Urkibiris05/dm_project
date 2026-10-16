"""Búsqueda por bloques con rutas optimizadas para distancias vectoriales."""

from typing import Self

import numpy as np

from ..distances import CosineDistance, Distance, MinkowskiDistance
from .base import NeighborSearch
from .blocked import BLOCK_BYTES


class OptimizedBlockedBruteForceSearch(NeighborSearch):
    """Búsqueda por bloques que evita trabajo innecesario al comparar radios.

    Para Euclídea compara distancias cuadradas con ``eps**2`` y para coseno
    normaliza una sola vez y compara productos escalares con ``1 - eps``.
    Las demás distancias usan la implementación genérica existente.

    ``memory_limit_mb`` limita la matriz temporal de distancias. No incluye la
    memoria de los vecindarios que DBSCAN conserva para expandir clusters.
    """

    def __init__(
        self,
        block_size: int | None = None,
        memory_limit_mb: float | None = None,
    ) -> None:
        if block_size is not None and block_size < 1:
            raise ValueError(
                f"block_size debe ser al menos 1 (block_size={block_size})"
            )
        if memory_limit_mb is not None and memory_limit_mb <= 0:
            raise ValueError(
                "memory_limit_mb debe ser mayor que 0 "
                f"(memory_limit_mb={memory_limit_mb})"
            )
        self.block_size = block_size
        self.memory_limit_mb = memory_limit_mb

    def fit(self, X: np.ndarray, distance: Distance) -> Self:
        self.X_ = np.asarray(X, dtype=float)
        self.distance_ = distance
        self.mode_ = self._mode(distance)

        if self.block_size is not None:
            self.block_size_ = self.block_size
        else:
            limit_bytes = (
                BLOCK_BYTES
                if self.memory_limit_mb is None
                else self.memory_limit_mb * 2**20
            )
            rows = int(limit_bytes) // (
                self.X_.dtype.itemsize * max(len(self.X_), 1)
            )
            self.block_size_ = max(1, rows)

        if self.mode_ == "cosine_normalized":
            norms = np.linalg.norm(self.X_, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.normalized_X_ = self.X_ / norms
        return self

    @staticmethod
    def _mode(distance: Distance) -> str:
        if isinstance(distance, MinkowskiDistance) and distance.p == 2:
            return "euclidean_squared"
        if isinstance(distance, CosineDistance):
            return "cosine_normalized"
        return "generic"

    def radius_neighbors(self, eps: float) -> list[np.ndarray]:
        neighborhoods: list[np.ndarray] = []
        threshold = eps * eps if self.mode_ == "euclidean_squared" else 1 - eps

        for start in range(0, len(self.X_), self.block_size_):
            block = self.X_[start:start + self.block_size_]
            if self.mode_ == "euclidean_squared":
                distances = self._squared_euclidean(block, self.X_)
                for row in distances:
                    neighborhoods.append(np.flatnonzero(row <= threshold))
            elif self.mode_ == "cosine_normalized":
                similarities = self.normalized_X_[start:start + len(block)] @ (
                    self.normalized_X_.T
                )
                for row in similarities:
                    neighborhoods.append(np.flatnonzero(row >= threshold))
            else:
                distances = self.distance_.pairwise(block, self.X_)
                for row in distances:
                    neighborhoods.append(np.flatnonzero(row <= eps))
        return neighborhoods

    @staticmethod
    def _squared_euclidean(
        points_a: np.ndarray, points_b: np.ndarray
    ) -> np.ndarray:
        squares_a = (points_a ** 2).sum(axis=1)[:, np.newaxis]
        squares_b = (points_b ** 2).sum(axis=1)[np.newaxis, :]
        squared = squares_a + squares_b - 2 * (points_a @ points_b.T)
        return np.maximum(squared, 0)
