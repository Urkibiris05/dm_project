"""E1.2: barrido de (eps, minPts) en cada espacio con métricas internas.

Para cada espacio y cada minPts, eps toma los cuantiles del k-dist y el codo.
Se ajusta nuestro DBSCAN y se guardan la estructura (clusters, ruido, cluster
mayor) y los índices internos. No se usa ninguna etiqueta Gold.
"""

import resource
import time

import numpy as np
import pandas as pd

from dbscan_implementation import DBSCAN
from evaluation.kdist import EPS_SHIFT, eps_grid, kdist, knee_point
from evaluation.metrics import internal_metrics, structure_summary
from evaluation.selection import add_validity

from . import config, datasets


def candidate_eps(
    kd: np.ndarray, quantiles: np.ndarray
) -> list[tuple[float, str, float]]:
    """Lista de ``(eps, origen, cuantil)``: los cuantiles y el codo."""
    candidates = [
        (float(eps), "quantile", float(q))
        for eps, q in zip(eps_grid(kd, quantiles), quantiles)
    ]
    _, knee = knee_point(kd)
    knee_q = float(np.mean(kd <= knee))
    candidates.append((knee * (1 + EPS_SHIFT), "knee", knee_q))
    return candidates


def fit_dbscan(
    X: np.ndarray, eps: float, min_samples: int
) -> tuple[np.ndarray, float]:
    """Ajusta nuestro DBSCAN y devuelve las etiquetas y los segundos."""
    model = DBSCAN(
        eps=eps, min_samples=min_samples, metric=config.METRIC,
        algorithm="optimized_blocked", deduplicate_queue=True)
    start = time.perf_counter()
    labels = model.fit_predict(X)
    return labels, time.perf_counter() - start


def peak_memory_mb() -> float:
    """Pico de memoria del proceso hasta ahora (Linux/WSL), en MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main(rapido: bool = False) -> None:
    rows: list[dict] = []
    quantiles = datasets.quantiles_for(rapido)
    for space in datasets.iter_spaces(rapido):
        X, _ = datasets.load_for(space, rapido)
        for min_samples in datasets.min_samples_for(space["unit"], rapido):
            kd = kdist(X, min_samples, config.METRIC)
            for eps, source, quantile in candidate_eps(kd, quantiles):
                labels, seconds = fit_dbscan(X, eps, min_samples)
                rows.append({
                    **space, "n_points": len(X),
                    "min_samples": min_samples, "eps": eps,
                    "eps_source": source, "eps_quantile": quantile,
                    **structure_summary(labels),
                    **internal_metrics(
                        X, labels, config.METRIC, config.DBCV_MAX_POINTS,
                        config.RANDOM_STATE),
                    "fit_seconds": seconds,
                    "peak_rss_mb": peak_memory_mb(),
                })
        print(f"   {space}", flush=True)
    grid = add_validity(pd.DataFrame(rows))
    config.RESULTS_DIR.mkdir(exist_ok=True)
    grid.to_csv(config.RESULTS_DIR / "e12_grid.csv", index=False)
