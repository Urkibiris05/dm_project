"""Búsquedas de vecinos disponibles para DBSCAN.

Para añadir una nueva:
  1. crear un archivo en esta carpeta con una clase que herede de
     ``NeighborSearch``;
  2. añadir su nombre a ``get_neighbor_search``.
"""

from .base import NeighborSearch
from .brute_force import BruteForceSearch

__all__ = ["NeighborSearch", "BruteForceSearch", "get_neighbor_search"]

# Nombres admitidos en el parámetro ``algorithm``.
ALGORITHM_NAMES: list[str] = ["brute"]


def get_neighbor_search(
    algorithm: str | NeighborSearch = "brute",
) -> NeighborSearch:
    """Devuelve el objeto ``NeighborSearch`` que corresponde a ``algorithm``.

    Parámetros
    ----------
    algorithm : str o NeighborSearch
        Nombre de la búsqueda, o directamente un objeto ``NeighborSearch``.
    """
    if isinstance(algorithm, NeighborSearch):
        return algorithm

    if algorithm == "brute":
        return BruteForceSearch()

    raise ValueError(
        f"Búsqueda desconocida: {algorithm!r}. Opciones: {ALGORITHM_NAMES}"
    )
