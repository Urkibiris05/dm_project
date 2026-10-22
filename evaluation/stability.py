"""Estabilidad de una configuración de DBSCAN: bootstrap y perturbación."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import adjusted_rand_score

from dbscan_implementation import DBSCAN


def _fit(
    X: np.ndarray, eps: float, min_samples: int, metric: str
) -> np.ndarray:
    """Ajusta nuestro DBSCAN (variante rápida, mismas etiquetas)."""
    model = DBSCAN(
        eps=eps, min_samples=min_samples, metric=metric,
        algorithm="optimized_blocked", deduplicate_queue=True)
    return model.fit_predict(X)


def bootstrap_stability(
    X: ArrayLike,
    eps: float,
    min_samples: int,
    metric: str = "cosine",
    n_runs: int = 20,
    frac: float = 0.8,
    seed: int = 0,
) -> dict:
    """ARI entre la partición completa y la de submuestras del ``frac``.

    En cada repetición se toma una submuestra sin reemplazo, se reajusta y
    se compara con las etiquetas de la partición completa sobre los puntos
    comunes. Devuelve la media, la desviación y el percentil 10 (un valor
    bajo del percentil delata resultados bimodales).
    """
    points = np.asarray(X, dtype=float)
    full = _fit(points, eps, min_samples, metric)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_runs):
        chosen = np.sort(rng.choice(
            len(points), size=int(frac * len(points)), replace=False))
        sub = _fit(points[chosen], eps, min_samples, metric)
        scores.append(adjusted_rand_score(full[chosen], sub))
    return {
        "stab_mean": float(np.mean(scores)),
        "stab_std": float(np.std(scores)),
        "stab_p10": float(np.percentile(scores, 10)),
    }


def eps_perturbation(
    X: ArrayLike,
    eps: float,
    min_samples: int,
    metric: str = "cosine",
    rel: float = 0.05,
) -> float:
    """Mínimo ARI entre la partición en eps y en eps * (1 -/+ rel).

    Un valor alto indica una meseta (resultado estable); uno bajo, que eps
    está en una transición de fase.
    """
    points = np.asarray(X, dtype=float)
    base = _fit(points, eps, min_samples, metric)
    scores = [
        adjusted_rand_score(
            base, _fit(points, eps * (1 + sign * rel), min_samples, metric))
        for sign in (-1, 1)
    ]
    return float(min(scores))
