"""Búsquedas de vecinos disponibles para DBSCAN.

Para añadir una nueva:
  1. crear un archivo en esta carpeta con una clase que herede de
     ``NeighborSearch``;
  2. añadir su nombre a ``get_neighbor_search``.
"""

from .base import NeighborSearch
from .blocked import BlockedBruteForceSearch
from .brute_force import BruteForceSearch
from .optimized_blocked import OptimizedBlockedBruteForceSearch

__all__ = [
    "NeighborSearch",
    "BruteForceSearch",
    "BlockedBruteForceSearch",
    "OptimizedBlockedBruteForceSearch",
    "get_neighbor_search",
]

# Nombres admitidos en el parámetro ``algorithm``.
ALGORITHM_NAMES: list[str] = ["brute", "blocked", "optimized_blocked"]


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
    if algorithm == "blocked":
        return BlockedBruteForceSearch()
    if algorithm == "optimized_blocked":
        return OptimizedBlockedBruteForceSearch()

    raise ValueError(
        f"Búsqueda desconocida: {algorithm!r}. Opciones: {ALGORITHM_NAMES}"
    )
