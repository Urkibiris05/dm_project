"""DBCV (Density-Based Clustering Validation), Moulavi et al. (2014).

Índice interno pensado para clusters de densidad con ruido. Va de -1 a 1:
valores altos indican clusters densos y bien separados. El ruido no
pertenece a ningún cluster, pero cuenta en el denominador N, así que
penaliza.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from dbscan_implementation.distances import get_distance

from . import NOISE

MIN_DISTANCE: float = 1e-12  # evita log(0) con puntos duplicados


def _core_distances(dist: np.ndarray, n_dims: int) -> np.ndarray:
    """Distancia núcleo de cada punto de un cluster.

    a(x) = ( sum_{y != x} (1 / d(x, y))^d / (n - 1) )^(-1/d). Se calcula en
    logaritmos porque con d grande (64-384) las potencias desbordan.
    """
    n_points = len(dist)
    log_terms = -n_dims * np.log(np.maximum(dist, MIN_DISTANCE))
    np.fill_diagonal(log_terms, -np.inf)
    log_mean = logsumexp(log_terms, axis=1) - np.log(n_points - 1)
    return np.exp(-log_mean / n_dims)


def _mutual_reachability(dist: np.ndarray, core: np.ndarray) -> np.ndarray:
    """m(x, y) = max(a(x), a(y), d(x, y))."""
    reach = np.maximum(dist, core[:, None])
    reach = np.maximum(reach, core[None, :])
    np.fill_diagonal(reach, 0)
    return reach


def _prim_mst(weights: np.ndarray) -> list[tuple[int, int, float]]:
    """Árbol de expansión mínimo (Prim, O(n^2)) de un grafo completo."""
    n_points = len(weights)
    in_tree = np.zeros(n_points, dtype=bool)
    in_tree[0] = True
    best = weights[0].copy()
    parent = np.zeros(n_points, dtype=int)
    edges = []
    for _ in range(n_points - 1):
        new = int(np.argmin(np.where(in_tree, np.inf, best)))
        edges.append((int(parent[new]), new, float(best[new])))
        in_tree[new] = True
        update = (weights[new] < best) & ~in_tree
        best[update] = weights[new][update]
        parent[update] = new
    return edges


def _internal_nodes(
    edges: list[tuple[int, int, float]], n_points: int
) -> np.ndarray:
    """Puntos con grado > 1 en el MST (si no hay, se usan todos)."""
    degree = np.zeros(n_points, dtype=int)
    for a, b, _ in edges:
        degree[a] += 1
        degree[b] += 1
    internal = degree > 1
    return internal if internal.any() else np.ones(n_points, dtype=bool)


def _subsample(
    points: np.ndarray, labels: np.ndarray, max_points: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Submuestra estratificada por etiqueta (el ruido es un estrato más)."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(points), dtype=bool)
    fraction = max_points / len(points)
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        size = max(1, int(round(len(members) * fraction)))
        keep[rng.choice(members, size=size, replace=False)] = True
    return points[keep], labels[keep]


def dbcv(
    X: ArrayLike,
    labels: ArrayLike,
    metric: str = "cosine",
    max_points: int | None = 3000,
    seed: int = 0,
    p: float | None = None,
) -> float:
    """Calcula el DBCV de una partición (el ruido es la etiqueta -1).

    Con ``max_points`` se submuestrea para acotar el coste O(n^2). Devuelve
    NaN con menos de 2 clusters o con clusters de un solo punto.
    """
    points = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if max_points is not None and len(points) > max_points:
        points, labels = _subsample(points, labels, max_points, seed)

    cluster_ids = [c for c in np.unique(labels) if c != NOISE]
    if len(cluster_ids) < 2:
        return float("nan")
    distance = get_distance(metric, p)
    n_dims = points.shape[1]

    # Por cluster: puntos, distancias núcleo, nodos internos y dispersión.
    info = {}
    for cluster in cluster_ids:
        members = np.flatnonzero(labels == cluster)
        if len(members) < 2:
            return float("nan")
        dist = distance.pairwise(points[members], points[members])
        core = _core_distances(dist, n_dims)
        edges = _prim_mst(_mutual_reachability(dist, core))
        internal = _internal_nodes(edges, len(members))
        inner = [w for a, b, w in edges if internal[a] and internal[b]]
        sparseness = max(inner) if inner else max(w for _, _, w in edges)
        info[cluster] = (members[internal], core[internal], sparseness,
                         len(members))

    # Separación: mínimo de m(x, y) entre nodos internos de dos clusters.
    separation = {c: np.inf for c in cluster_ids}
    for i, ci in enumerate(cluster_ids):
        nodes_i, core_i = info[ci][0], info[ci][1]
        for cj in cluster_ids[i + 1:]:
            nodes_j, core_j = info[cj][0], info[cj][1]
            dist = distance.pairwise(points[nodes_i], points[nodes_j])
            reach = np.maximum(dist, core_i[:, None])
            reach = np.maximum(reach, core_j[None, :])
            value = float(reach.min())
            separation[ci] = min(separation[ci], value)
            separation[cj] = min(separation[cj], value)

    score = 0.0
    for cluster in cluster_ids:
        sparseness, size = info[cluster][2], info[cluster][3]
        dspc = separation[cluster]
        validity = (dspc - sparseness) / max(dspc, sparseness)
        score += size / len(points) * validity
    return float(score)
