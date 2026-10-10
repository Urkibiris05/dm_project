"""Distancia coseno."""

import numpy as np
from numpy.typing import ArrayLike

from .base import Distance


class CosineDistance(Distance):
    """Distancia coseno: ``d(a, b) = 1 - cos(a, b)``.

    Vale 0 si los vectores apuntan en la misma dirección, 1 si son
    ortogonales y 2 si son opuestos. No depende de la norma de los vectores.

    Un vector nulo no tiene dirección; por convenio su distancia a cualquier
    otro vector (incluido él mismo) es 1.
    """

    def pairwise(self, points_a: ArrayLike, points_b: ArrayLike) -> np.ndarray:
        points_a = np.asarray(points_a, dtype=float)
        points_b = np.asarray(points_b, dtype=float)

        # cos(a, b) = (a . b) / (|a| |b|): se normaliza cada fila y después
        # un producto matricial da todos los cosenos a la vez.
        unit_a = self._normalize(points_a)
        unit_b = self._normalize(points_b)
        distances = 1 - unit_a @ unit_b.T

        # Los errores de redondeo pueden dar valores como -1e-16.
        return np.clip(distances, 0, 2)

    @staticmethod
    def _normalize(points: np.ndarray) -> np.ndarray:
        """Divide cada fila por su norma (las filas nulas se dejan a cero)."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return points / norms

    def __repr__(self) -> str:
        return "CosineDistance()"
