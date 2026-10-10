"""Clase base de la que heredan todas las distancias."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


class Distance(ABC):
    """Una distancia entre vectores.

    Para añadir una distancia nueva basta con heredar de esta clase,
    implementar ``pairwise`` y registrarla en ``distances/__init__.py``.
    """

    @abstractmethod
    def pairwise(self, points_a: ArrayLike, points_b: ArrayLike) -> np.ndarray:
        """Distancias entre las filas de ``points_a`` y las de ``points_b``.

        Parámetros
        ----------
        points_a : array de forma (n_a, n_dimensiones)
        points_b : array de forma (n_b, n_dimensiones)

        Devuelve
        --------
        distances : array de forma (n_a, n_b), donde ``distances[i, j]`` es
            la distancia entre ``points_a[i]`` y ``points_b[j]``.
        """
