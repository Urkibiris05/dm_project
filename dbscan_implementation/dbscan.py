"""Algoritmo DBSCAN (Ester et al., 1996) con la interfaz de scikit-learn."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from .distances import Distance, get_distance
from .neighbors import NeighborSearch, get_neighbor_search

NOISE: int = -1


class DBSCAN:
    """Clustering por densidad.

    Un punto es *núcleo* si tiene al menos ``min_samples`` puntos (contándose
    a sí mismo) a distancia ``<= eps``. Un cluster es un conjunto de puntos
    núcleo conectados entre sí, más los puntos *frontera* que caen dentro del
    radio de alguno de ellos. Lo que no pertenece a ningún cluster es ruido.

    Parámetros
    ----------
    eps : float
        Radio del vecindario.
    min_samples : int
        Número mínimo de puntos en el vecindario (incluido el propio punto)
        para que un punto sea núcleo.
    metric : str o Distance
        ``"euclidean"``, ``"manhattan"``, ``"chebyshev"``, ``"minkowski"``
        o ``"cosine"``; o un objeto ``Distance`` propio.
    p : float, opcional
        Exponente de Minkowski. Solo se usa con ``metric="minkowski"``.
    algorithm : str o NeighborSearch
        Cómo se buscan los vecinos. De momento solo ``"brute"``; también se
        puede pasar un objeto ``NeighborSearch`` propio.

    Atributos (disponibles después de ``fit``)
    ------------------------------------------
    labels_ : array de forma (n_puntos,)
        Cluster de cada punto (0, 1, 2...). El ruido se marca con -1.
    core_sample_indices_ : array
        Índices de los puntos núcleo.
    components_ : array de forma (n_nucleos, n_dimensiones)
        Copia de los puntos núcleo.

    Ejemplo
    -------
    >>> from dbscan_implementation import DBSCAN
    >>> X = [[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]
    >>> DBSCAN(eps=3, min_samples=2).fit_predict(X)
    array([ 0,  0,  0,  1,  1, -1])
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str | Distance = "euclidean",
        p: float | None = None,
        algorithm: str | NeighborSearch = "brute",
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.p = p
        self.algorithm = algorithm

    def fit(self, X: ArrayLike) -> Self:
        """Agrupa los puntos de ``X`` y guarda el resultado en ``labels_``."""
        if self.eps <= 0:
            raise ValueError(f"eps debe ser mayor que 0 (eps={self.eps})")
        if self.min_samples < 1:
            raise ValueError(
                "min_samples debe ser al menos 1 "
                f"(min_samples={self.min_samples})"
            )

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                "X debe ser una tabla de forma (n_puntos, n_dimensiones)"
            )

        # Paso 1: vecindario de cada punto.
        distance = get_distance(self.metric, self.p)
        search = get_neighbor_search(self.algorithm)
        neighborhoods = search.fit(X, distance).radius_neighbors(self.eps)

        # Paso 2: puntos núcleo.
        neighbor_counts = np.array([len(n) for n in neighborhoods])
        is_core = neighbor_counts >= self.min_samples

        # Paso 3: formar los clusters.
        self.labels_: np.ndarray = self._expand_clusters(
            neighborhoods, is_core
        )
        self.core_sample_indices_: np.ndarray = np.flatnonzero(is_core)
        self.components_: np.ndarray = X[self.core_sample_indices_].copy()
        return self

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        """Agrupa los puntos de ``X`` y devuelve la etiqueta de cada uno."""
        return self.fit(X).labels_

    @staticmethod
    def _expand_clusters(
        neighborhoods: list[np.ndarray],
        is_core: np.ndarray,
    ) -> np.ndarray:
        """Asigna una etiqueta a cada punto a partir de los vecindarios.

        Se recorren los puntos en orden. Cada punto núcleo que todavía no
        tiene cluster inicia uno nuevo, que se extiende a sus vecinos, a los
        vecinos de sus vecinos núcleo, y así sucesivamente.
        """
        n_points = len(neighborhoods)
        labels = np.full(n_points, NOISE, dtype=int)
        cluster = 0

        for start in range(n_points):
            if labels[start] != NOISE or not is_core[start]:
                continue

            # Puntos pendientes de visitar dentro del cluster actual.
            pending = [start]
            while pending:
                point = pending.pop()
                if labels[point] != NOISE:
                    continue  # ya tiene cluster (este u otro anterior)
                labels[point] = cluster

                # Solo los puntos núcleo extienden el cluster. Un punto
                # frontera se queda en el cluster, pero no arrastra a sus
                # vecinos.
                if is_core[point]:
                    for neighbor in neighborhoods[point]:
                        if labels[neighbor] == NOISE:
                            pending.append(neighbor)

            cluster += 1

        return labels
