"""E0.1: nuestras variantes de DBSCAN frente a scikit-learn.

Para cada dataset, métrica, min_samples y eps se comprueba si las etiquetas
de cada variante son idénticas a las de sklearn.cluster.DBSCAN.
"""

import time

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN as SklearnDBSCAN

from dbscan_implementation import DBSCAN

from . import analysis
from .config import (
    EPS_QUANTILES,
    METRICS,
    MIN_SAMPLES_GRID,
    OUR_VARIANTS,
    RESULTS_DIR,
)
from .datasets import distance_matrix, eps_grid, kdist_from_matrix
from .datasets import load_datasets

# Caso cuyas particiones se guardan para la figura 2D.
FIGURE_MIN_SAMPLES: int = 5
FIGURE_QUANTILE: float = 0.7
FIGURE_VARIANTS: tuple[str, ...] = ("ours_brute", "ours_optimized_blocked")
TWO_D: tuple[str, ...] = (
    "blobs", "varied", "aniso", "moons", "circles", "no_structure",
)


def save_kdist(datasets: dict) -> None:
    """Guarda las curvas k-dist ordenadas (euclídea, min_samples = 5)."""
    rows = []
    for name, dataset in datasets.items():
        if name == "emb_384d":
            continue
        D = distance_matrix(dataset.X, "euclidean")
        values = np.sort(kdist_from_matrix(D, FIGURE_MIN_SAMPLES))
        for rank, value in enumerate(values):
            rows.append({"dataset": name, "rank": rank, "kdist": value})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "e00_kdist.csv", index=False)


def main(rapido: bool = False) -> None:
    datasets = load_datasets()
    save_kdist(datasets)
    min_samples_grid = (5,) if rapido else MIN_SAMPLES_GRID
    quantiles = EPS_QUANTILES[[2, 4, 6]] if rapido else EPS_QUANTILES

    rows: list[dict] = []
    figure_rows: list[dict] = []
    for name, dataset in datasets.items():
        if name == "emb_384d":
            continue
        X = dataset.X
        for metric, p in METRICS:
            D = distance_matrix(X, metric, p)
            for min_samples in min_samples_grid:
                eps_values = eps_grid(D, min_samples, quantiles)
                for quantile, eps in zip(quantiles, eps_values):
                    eps = float(eps)
                    kwargs = {"p": p} if metric == "minkowski" else {}
                    reference = SklearnDBSCAN(
                        eps=eps, min_samples=min_samples, metric=metric,
                        algorithm="brute", **kwargs,
                    ).fit(X)
                    neigh = analysis.neighborhoods_reference(
                        X, eps, metric, p
                    )
                    core = analysis.core_mask(neigh, min_samples)
                    boundary = analysis.n_boundary_pairs(D, eps)
                    labels_by_variant = {}
                    for variant, extra in OUR_VARIANTS.items():
                        start = time.perf_counter()
                        ours = DBSCAN(
                            eps=eps, min_samples=min_samples,
                            metric=metric, p=p, **extra,
                        ).fit(X)
                        seconds = time.perf_counter() - start
                        labels_by_variant[variant] = ours.labels_
                        comparison = analysis.compare_partitions(
                            reference.labels_, ours.labels_, core
                        )
                        summary = analysis.summarize(ours.labels_)
                        rows.append({
                            "dataset": name,
                            "n": len(X),
                            "d": X.shape[1],
                            "metric": metric,
                            "p": p,
                            "min_samples": min_samples,
                            "eps_quantile": float(quantile),
                            "eps": eps,
                            "variant": variant,
                            "exact_equal": comparison["exact_equal"],
                            "same_core_indices": bool(np.array_equal(
                                ours.core_sample_indices_,
                                reference.core_sample_indices_,
                            )),
                            "ari": comparison["ari"],
                            "ami": comparison["ami"],
                            "same_noise": comparison["same_noise"],
                            "n_diff_core": comparison["n_diff_core"],
                            "n_diff_border": comparison["n_diff_border"],
                            "n_diff_noise": comparison["n_diff_noise"],
                            "n_clusters_ours": summary["n_clusters"],
                            "n_clusters_sklearn": analysis.summarize(
                                reference.labels_
                            )["n_clusters"],
                            "noise_rate": summary["noise_rate"],
                            "n_boundary_pairs": boundary,
                            "seconds": seconds,
                        })
                    is_figure_case = (
                        name in TWO_D
                        and metric == "euclidean"
                        and min_samples == FIGURE_MIN_SAMPLES
                        and np.isclose(quantile, FIGURE_QUANTILE)
                    )
                    if is_figure_case:
                        for i in range(len(X)):
                            row = {
                                "dataset": name, "x": X[i, 0], "y": X[i, 1],
                                "is_core": bool(core[i]),
                                "sklearn": int(reference.labels_[i]),
                            }
                            for variant in FIGURE_VARIANTS:
                                row[variant] = int(
                                    labels_by_variant[variant][i]
                                )
                            figure_rows.append(row)
        print(f"  E0.1 {name} listo")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "e01_vs_sklearn.csv", index=False)
    pd.DataFrame(figure_rows).to_csv(
        RESULTS_DIR / "e01_particiones_2d.csv", index=False
    )

    summary = table.groupby(["variant", "metric"]).agg(
        n_casos=("exact_equal", "size"),
        pct_exact_equal=("exact_equal", lambda s: 100 * s.mean()),
        ari_min=("ari", "min"),
        casos_con_empate_frontera=(
            "n_boundary_pairs", lambda s: int((s > 0).sum())
        ),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "e01_resumen.csv", index=False)

    discrepancies = table[~table["exact_equal"]]
    discrepancies.to_csv(RESULTS_DIR / "e01_discrepancias.csv", index=False)
    print(f"  E0.1: {len(table)} comparaciones, "
          f"{len(discrepancies)} discrepancias")
