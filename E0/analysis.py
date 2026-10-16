"""Funciones para comparar particiones y comprobar que son DBSCAN válidos."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from .config import BOUNDARY_TOL

NOISE: int = -1


def neighborhoods_reference(
    X: np.ndarray, eps: float, metric: str, p: float | None = None
) -> list[np.ndarray]:
    """Vecindarios calculados con scikit-learn (independientes de los
    nuestros). Cada vecindario incluye al propio punto."""
    kwargs = {"p": p} if metric == "minkowski" else {}
    search = NearestNeighbors(
        radius=eps, metric=metric, algorithm="brute", **kwargs
    ).fit(X)
    return list(search.radius_neighbors(X, return_distance=False))


def core_mask(neigh: list[np.ndarray], min_samples: int) -> np.ndarray:
    """Marca los puntos núcleo (al menos min_samples vecinos)."""
    return np.array([len(n) >= min_samples for n in neigh])


def n_boundary_pairs(D: np.ndarray, eps: float) -> int:
    """Pares i < j cuya distancia empata con eps (|d - eps| <= tol*eps)."""
    close = np.abs(D - eps) <= BOUNDARY_TOL * eps
    return int(np.triu(close, k=1).sum())


def summarize(labels: np.ndarray) -> dict:
    """Número de clusters y tasa de ruido de una partición."""
    return {
        "n_clusters": len(set(labels.tolist()) - {NOISE}),
        "noise_rate": float(np.mean(labels == NOISE)),
    }


def _match_clusters(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Renombra los clusters de b para que coincidan con los de a.

    Usa el emparejamiento que maximiza las coincidencias (algoritmo
    húngaro). El ruido solo se empareja con el ruido; los clusters de b
    sin pareja reciben la etiqueta -2.
    """
    ids_a = sorted(set(a.tolist()) - {NOISE})
    ids_b = sorted(set(b.tolist()) - {NOISE})
    mapped = np.where(b == NOISE, NOISE, -2)
    if not ids_a or not ids_b:
        return mapped
    table = np.zeros((len(ids_a), len(ids_b)), dtype=int)
    for i, label_a in enumerate(ids_a):
        for j, label_b in enumerate(ids_b):
            table[i, j] = np.sum((a == label_a) & (b == label_b))
    rows, cols = linear_sum_assignment(table, maximize=True)
    for i, j in zip(rows, cols):
        mapped[b == ids_b[j]] = ids_a[i]
    return mapped


def compare_partitions(
    a: np.ndarray, b: np.ndarray, core: np.ndarray
) -> dict:
    """Compara dos particiones (a es la de referencia).

    n_diff_noise: puntos que uno marca como ruido y el otro no.
    Del resto de diferencias, n_diff_core son puntos núcleo y
    n_diff_border el resto (fronteras).
    """
    mapped = _match_clusters(a, b)
    different = a != mapped
    noise_diff = different & ((a == NOISE) | (b == NOISE))
    other = different & ~noise_diff
    if core.sum() >= 2:
        core_ari = adjusted_rand_score(a[core], b[core])
    else:
        core_ari = 1.0
    return {
        "exact_equal": bool(np.array_equal(a, b)),
        "ari": adjusted_rand_score(a, b),
        "ami": adjusted_mutual_info_score(a, b),
        "same_noise": bool(np.array_equal(a == NOISE, b == NOISE)),
        "core_ari": core_ari,
        "n_diff_core": int(np.sum(other & core)),
        "n_diff_border": int(np.sum(other & ~core)),
        "n_diff_noise": int(noise_diff.sum()),
    }


def dbscan_validity(
    labels: np.ndarray, neigh: list[np.ndarray], core: np.ndarray
) -> dict:
    """Comprueba que labels es una salida válida de DBSCAN.

    Vale para cualquier orden de recorrido. n_ambiguous_border cuenta
    las fronteras con vecinos núcleo de varios clusters: su etiqueta depende
    del orden, es la indeterminación conocida de DBSCAN.
    """
    cores_labelled = bool(np.all(labels[core] != NOISE))
    noise_correct = True
    border_valid = True
    n_ambiguous = 0
    for i, neighbors in enumerate(neigh):
        if core[i]:
            continue
        core_neighbors = neighbors[core[neighbors]]
        reachable = len(core_neighbors) > 0
        if (labels[i] == NOISE) == reachable:
            noise_correct = False
        if not reachable:
            continue
        near_labels = set(labels[core_neighbors].tolist())
        if labels[i] != NOISE and labels[i] not in near_labels:
            border_valid = False
        if len(near_labels) >= 2:
            n_ambiguous += 1
    return {
        "cores_labelled": cores_labelled,
        "noise_correct": noise_correct,
        "border_valid": border_valid,
        "n_ambiguous_border": n_ambiguous,
    }
