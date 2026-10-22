"""Grafo k-dist, codo automático y diagnóstico de concentración."""

import numpy as np
from numpy.typing import ArrayLike

from dbscan_implementation.distances import get_distance

# Desplazamiento relativo de eps para no caer en empates exactos (ver E0).
EPS_SHIFT: float = 1e-7


def kdist(
    X: ArrayLike,
    k: int,
    metric: str = "cosine",
    block_size: int = 1024,
) -> np.ndarray:
    """Distancia de cada punto a su k-ésimo vecino, contando el propio punto.

    Es la convención de nuestro DBSCAN: un punto es núcleo si tiene al menos
    ``minPts`` puntos a distancia <= eps (él incluido), así que se usa
    ``k = minPts``. Se calcula por bloques de filas para no guardar la matriz
    n x n completa.
    """
    points = np.asarray(X, dtype=float)
    n_points = len(points)
    if not 1 <= k <= n_points:
        raise ValueError(f"k debe estar entre 1 y {n_points}, no {k}")
    distance = get_distance(metric)
    result = np.empty(n_points)
    for start in range(0, n_points, block_size):
        block = distance.pairwise(points[start:start + block_size], points)
        result[start:start + block_size] = np.partition(
            block, k - 1, axis=1)[:, k - 1]
    return result


def _chord_gap(values: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Curva ordenada y su distancia (normalizada) a la cuerda."""
    curve = np.sort(np.asarray(values, dtype=float))
    if len(curve) < 3 or curve[-1] == curve[0]:
        return curve, np.zeros(len(curve))
    x = np.linspace(0, 1, len(curve))
    y = (curve - curve[0]) / (curve[-1] - curve[0])
    # La cuerda es la recta y = x; la distancia es proporcional a |x - y|.
    return curve, np.abs(x - y)


def knee_point(values: ArrayLike) -> tuple[int, float]:
    """Codo de una curva: punto más alejado de la cuerda primero-último.

    Se ordenan los valores de menor a mayor y se normalizan ambos ejes a
    [0, 1]. Devuelve la posición en la curva ordenada y el valor (eps).
    """
    curve, gap = _chord_gap(values)
    index = int(np.argmax(gap))
    return index, float(curve[index])


def knee_sharpness(values: ArrayLike) -> float:
    """Distancia máxima a la cuerda (0 = curva recta, mayor = codo claro)."""
    _, gap = _chord_gap(values)
    return float(gap.max() / np.sqrt(2)) if len(gap) else 0.0


def relative_contrast(
    X: ArrayLike,
    metric: str = "cosine",
    n_queries: int = 1000,
    seed: int = 0,
) -> float:
    """Mediana de (D_max - D_min) / D_min sobre consultas aleatorias.

    Mide la concentración de distancias (sección 5.3 del estado del arte):
    cuanto más cerca de 0, menos contraste entre vecino cercano y lejano.
    El propio punto y las distancias nulas (duplicados) se excluyen.
    """
    points = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    n_queries = min(n_queries, len(points))
    queries = rng.choice(len(points), size=n_queries, replace=False)
    distances = get_distance(metric).pairwise(points[queries], points)
    contrasts = []
    for row in distances:
        positive = row[row > 1e-12]
        if len(positive) >= 2:
            contrasts.append((positive.max() - positive.min())
                             / positive.min())
    return float(np.median(contrasts)) if contrasts else float("nan")


def eps_grid(kd: ArrayLike, quantiles: ArrayLike) -> np.ndarray:
    """Valores de eps: cuantiles del k-dist, desplazados un 1e-7 relativo."""
    values = np.quantile(np.asarray(kd, dtype=float), quantiles)
    return values * (1 + EPS_SHIFT)
