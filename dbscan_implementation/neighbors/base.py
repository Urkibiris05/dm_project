"""Clase base de la que heredan todas las búsquedas de vecinos."""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from ..distances import Distance


class NeighborSearch(ABC):
    """Estrategia para encontrar los vecinos de cada punto.

    DBSCAN solo necesita saber, para cada punto, qué puntos tiene a
    distancia ``<= eps``. Cómo se calcula eso (fuerza bruta, por bloques,
    con un índice...) es lo que decide cada subclase.

    Para añadir una búsqueda nueva basta con heredar de esta clase,
    implementar los dos métodos y registrarla en ``neighbors/__init__.py``.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, distance: Distance) -> Self:
        """Prepara la búsqueda sobre los datos ``X``.

        Parámetros
        ----------
        X : array de forma (n_puntos, n_dimensiones)
        distance : Distance
            Distancia con la que se comparan los puntos.

        Devuelve
        --------
        self
        """

    @abstractmethod
    def radius_neighbors(self, eps: float) -> list[np.ndarray]:
        """Vecinos de cada punto dentro del radio ``eps``.

        Devuelve
        --------
        neighborhoods : lista de n_puntos arrays de enteros
            ``neighborhoods[i]`` contiene los índices de los puntos a
            distancia ``<= eps`` del punto ``i``, ordenados de menor a
            mayor. El propio punto ``i`` está incluido.
        """
