"""Métricas de estructura, internas y externas de una partición DBSCAN."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             calinski_harabasz_score, completeness_score,
                             davies_bouldin_score, homogeneity_score,
                             silhouette_score)

from . import NOISE
from .dbcv import _subsample, dbcv


def structure_summary(labels: ArrayLike) -> dict:
    """Número de clusters, ruido y tamaños (fracciones sobre todos los puntos).

    ``largest_cluster_frac`` cerca de 1 delata el fallo típico de una mala
    parametrización: un cluster que lo absorbe todo.
    """
    labels = np.asarray(labels)
    ids, counts = np.unique(labels[labels != NOISE], return_counts=True)
    return {
        "n_clusters": int(len(ids)),
        "noise_rate": float(np.mean(labels == NOISE)),
        "largest_cluster_frac": (
            float(counts.max() / len(labels)) if len(ids) else 0.0),
        "min_cluster_size": int(counts.min()) if len(ids) else 0,
        "median_cluster_size": (
            float(np.median(counts)) if len(ids) else 0.0),
    }


def internal_metrics(
    X: ArrayLike,
    labels: ArrayLike,
    metric: str = "cosine",
    sample_size: int = 3000,
    seed: int = 0,
) -> dict:
    """DBCV (con todos los puntos) y otros índices internos sin el ruido.

    Silhouette, Davies-Bouldin y Calinski-Harabasz no saben tratar el ruido,
    así que se calculan solo con los puntos que pertenecen a un cluster. Se
    devuelve NaN si hay menos de 2 clusters.
    """
    points = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    result = {
        "dbcv": dbcv(points, labels, metric, sample_size, seed),
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
        "calinski_harabasz": float("nan"),
    }
    mask = labels != NOISE
    clustered, clustered_labels = points[mask], labels[mask]
    n_clusters = len(np.unique(clustered_labels))
    if n_clusters < 2 or len(clustered) <= n_clusters:
        return result
    sample, sample_labels = clustered, clustered_labels
    if len(clustered) > sample_size:
        # Submuestra estratificada: así no desaparece ningún cluster.
        sample, sample_labels = _subsample(
            clustered, clustered_labels, sample_size, seed)
    if len(np.unique(sample_labels)) >= 2 and len(sample) > len(
            np.unique(sample_labels)):
        result["silhouette"] = float(silhouette_score(
            sample, sample_labels, metric=metric))
    result["davies_bouldin"] = float(
        davies_bouldin_score(clustered, clustered_labels))
    result["calinski_harabasz"] = float(
        calinski_harabasz_score(clustered, clustered_labels))
    return result


def purity(labels: np.ndarray, y: np.ndarray) -> float:
    """Pureza: fracción de puntos de la clase mayoritaria de su cluster."""
    if len(labels) == 0:
        return float("nan")
    total = 0
    for cluster in np.unique(labels):
        _, counts = np.unique(y[labels == cluster], return_counts=True)
        total += counts.max()
    return float(total / len(labels))


def _external(labels: np.ndarray, y: np.ndarray, suffix: str) -> dict:
    """Métricas externas de un par de vectores, con el sufijo indicado."""
    if len(labels) < 2:
        return {f"{name}_{suffix}": float("nan") for name in
                ("ari", "ami", "homogeneity", "completeness", "purity")}
    return {
        f"ari_{suffix}": float(adjusted_rand_score(y, labels)),
        f"ami_{suffix}": float(adjusted_mutual_info_score(y, labels)),
        f"homogeneity_{suffix}": float(homogeneity_score(y, labels)),
        f"completeness_{suffix}": float(completeness_score(y, labels)),
        f"purity_{suffix}": purity(labels, y),
    }


def external_metrics(labels: ArrayLike, y: ArrayLike) -> dict:
    """ARI, AMI, homogeneidad, completitud y pureza frente a etiquetas Gold.

    ``*_all`` trata el ruido como un cluster más; ``*_nonoise`` lo excluye.
    SOLO PARA EVALUAR: nunca se usa para elegir parámetros (fuga de
    etiquetas). Las etiquetas son por usuario y el acuerdo entre anotadores
    es moderado, lo que limita el valor máximo alcanzable.
    """
    labels = np.asarray(labels)
    y = np.asarray(y)
    result = _external(labels, y, "all")
    mask = labels != NOISE
    result.update(_external(labels[mask], y[mask], "nonoise"))
    return result
