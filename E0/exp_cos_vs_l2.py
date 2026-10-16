"""E0.2: coseno frente a euclídea sobre vectores de norma 1.

Si |x| = |y| = 1 entonces |x - y|^2 = 2 * d_cos(x, y), así que DBSCAN con
eps_l2 = sqrt(2 * eps_cos) debe dar exactamente la misma partición.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN as SklearnDBSCAN

from dbscan_implementation import DBSCAN

from . import analysis
from .config import (
    EPS_QUANTILES,
    MIN_SAMPLES_GRID,
    OUR_VARIANTS,
    RANDOM_STATE,
    RESULTS_DIR,
)
from .datasets import distance_matrix, eps_grid, load_datasets
from .datasets import normalize_rows

# Control negativo: sin normalizar la equivalencia no tiene por qué cumplirse.
CONTROL_DATASETS: tuple[str, ...] = (
    "blobs", "varied", "aniso", "moons", "circles", "no_structure", "wine",
)
CONTROL_IMPLEMENTATIONS: tuple[str, ...] = ("ours_brute", "sklearn_brute")


def fit_labels(
    implementation: str, X: np.ndarray, eps: float, min_samples: int,
    metric: str,
) -> np.ndarray:
    """Etiquetas de una implementación (nuestra o sklearn_brute)."""
    if implementation == "sklearn_brute":
        model = SklearnDBSCAN(
            eps=eps, min_samples=min_samples, metric=metric,
            algorithm="brute",
        )
    else:
        model = DBSCAN(
            eps=eps, min_samples=min_samples, metric=metric,
            **OUR_VARIANTS[implementation],
        )
    return model.fit(X).labels_


def identity_table(datasets: dict) -> pd.DataFrame:
    """Compara d_l2^2 con 2 * d_cos en 2000 pares aleatorios."""
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for name in ("emb_384d", "moons"):
        Xn = normalize_rows(datasets[name].X)
        i = rng.integers(0, len(Xn), 2000)
        j = rng.integers(0, len(Xn), 2000)
        d_l2_sq = np.sum((Xn[i] - Xn[j]) ** 2, axis=1)
        d_cos = 1 - np.sum(Xn[i] * Xn[j], axis=1)
        for a, b in zip(d_l2_sq, 2 * d_cos):
            rows.append({
                "dataset": name, "d_l2_squared": a, "two_d_cos": b,
                "abs_error": abs(a - b),
            })
    return pd.DataFrame(rows)


def main(rapido: bool = False) -> None:
    datasets = load_datasets()
    min_samples_grid = (5,) if rapido else MIN_SAMPLES_GRID
    quantiles = EPS_QUANTILES[[2, 4, 6]] if rapido else EPS_QUANTILES
    implementations = list(OUR_VARIANTS) + ["sklearn_brute"]

    rows: list[dict] = []
    cases = [(n, True) for n in datasets]
    cases += [(n, False) for n in CONTROL_DATASETS]
    for name, normalized in cases:
        X = datasets[name].X
        if normalized:
            X = normalize_rows(X)
        impls = implementations if normalized else CONTROL_IMPLEMENTATIONS
        D_cos = distance_matrix(X, "cosine")
        everyone = np.ones(len(X), dtype=bool)
        for min_samples in min_samples_grid:
            eps_values = eps_grid(D_cos, min_samples, quantiles)
            for quantile, eps_cos in zip(quantiles, eps_values):
                eps_cos = float(eps_cos)
                eps_l2 = float(np.sqrt(2 * eps_cos))
                boundary = analysis.n_boundary_pairs(D_cos, eps_cos)
                for implementation in impls:
                    labels_cos = fit_labels(
                        implementation, X, eps_cos, min_samples, "cosine"
                    )
                    labels_l2 = fit_labels(
                        implementation, X, eps_l2, min_samples, "euclidean"
                    )
                    summary_cos = analysis.summarize(labels_cos)
                    summary_l2 = analysis.summarize(labels_l2)
                    rows.append({
                        "dataset": name,
                        "normalized": normalized,
                        "min_samples": min_samples,
                        "eps_quantile": float(quantile),
                        "eps_cos": eps_cos,
                        "eps_l2": eps_l2,
                        "implementation": implementation,
                        "exact_equal": bool(
                            np.array_equal(labels_cos, labels_l2)
                        ),
                        "ari": analysis.compare_partitions(
                            labels_cos, labels_l2, everyone
                        )["ari"],
                        "n_clusters_cos": summary_cos["n_clusters"],
                        "n_clusters_l2": summary_l2["n_clusters"],
                        "noise_rate_cos": summary_cos["noise_rate"],
                        "noise_rate_l2": summary_l2["noise_rate"],
                        "n_boundary_pairs": boundary,
                    })
        print(f"  E0.2 {name} (normalizado={normalized}) listo")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "e02_cos_vs_l2.csv", index=False)
    summary = table.groupby(["implementation", "normalized"]).agg(
        n_casos=("exact_equal", "size"),
        pct_exact_equal=("exact_equal", lambda s: 100 * s.mean()),
        ari_min=("ari", "min"),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "e02_resumen.csv", index=False)

    identity = identity_table(datasets)
    identity.to_csv(
        RESULTS_DIR / "e02_identidad_distancias.csv", index=False
    )
    print(f"  E0.2: error máximo de la identidad = "
          f"{identity['abs_error'].max():.2e}")
