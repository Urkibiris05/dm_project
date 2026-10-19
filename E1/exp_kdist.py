"""E1.1: diagnóstico de densidad (k-dist, codo y contraste de distancias).

No ajusta DBSCAN. Responde: ¿la SVD crea contraste de distancias? ¿Qué
analizador (palabra o carácter) da un codo más claro en el k-dist? Incluye
el control "sin SVD" (TF-IDF normalizado y disperso).
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors

from evaluation.kdist import (kdist, knee_point, knee_sharpness,
                              relative_contrast)

from . import config, datasets

NO_SVD: int = 0  # valor de la columna svd para el control sin SVD


def sparse_kdist(matrix: sparse.csr_matrix, k: int) -> np.ndarray:
    """k-dist (propio punto incluido) con coseno sobre una matriz dispersa."""
    search = NearestNeighbors(
        n_neighbors=k, metric="cosine", algorithm="brute").fit(matrix)
    return search.kneighbors(matrix)[0][:, k - 1]


def sparse_contrast(
    matrix: sparse.csr_matrix, n_queries: int = 500, seed: int = 0
) -> float:
    """Contraste relativo mediano para una matriz dispersa."""
    rng = np.random.default_rng(seed)
    queries = rng.choice(
        matrix.shape[0], size=min(n_queries, matrix.shape[0]), replace=False)
    distances = cosine_distances(matrix[queries], matrix)
    values = []
    for row in distances:
        positive = row[row > 1e-12]
        if len(positive) >= 2:
            values.append((positive.max() - positive.min()) / positive.min())
    return float(np.median(values)) if values else float("nan")


def describe(
    space: dict, min_samples: int, kd: np.ndarray, contrast: float,
    curves: list[pd.DataFrame],
) -> dict:
    """Fila del diagnóstico y curva k-dist (submuestreada) de un caso."""
    _, eps_knee = knee_point(kd)
    ordered = np.sort(kd)
    positions = np.linspace(0, len(ordered) - 1, config.KDIST_POINTS)
    curves.append(pd.DataFrame({
        **space, "min_samples": min_samples,
        "quantile": positions / (len(ordered) - 1),
        "kdist": ordered[positions.astype(int)],
    }))
    return {
        **space, "min_samples": min_samples, "n_points": len(kd),
        "eps_knee": eps_knee, "knee_sharpness": knee_sharpness(kd),
        "kdist_median": float(np.median(kd)),
        "kdist_p10": float(np.quantile(kd, 0.1)),
        "kdist_p90": float(np.quantile(kd, 0.9)),
        "relative_contrast": contrast,
    }


def subsample_sparse(
    matrix: sparse.csr_matrix, n_rows: int
) -> sparse.csr_matrix:
    """Submuestra de filas reproducible (modo rápido)."""
    if matrix.shape[0] <= n_rows:
        return matrix
    rows = np.sort(np.random.default_rng(config.RANDOM_STATE).choice(
        matrix.shape[0], size=n_rows, replace=False))
    return matrix[rows]


def main(rapido: bool = False) -> None:
    rows: list[dict] = []
    curves: list[pd.DataFrame] = []
    for space in datasets.iter_spaces(rapido):
        X, _ = datasets.load_for(space, rapido)
        contrast = relative_contrast(
            X, config.METRIC, seed=config.RANDOM_STATE)
        for min_samples in datasets.min_samples_for(space["unit"], rapido):
            kd = kdist(X, min_samples, config.METRIC)
            rows.append(describe(space, min_samples, kd, contrast, curves))

    # Control sin SVD: se parte del vocabulario del espacio con SVD de 128
    # (64 en modo rápido), pero se usa la matriz TF-IDF dispersa.
    control_svd = config.RAPIDO_SVD[0] if rapido else 128
    disorders = config.RAPIDO_DISORDERS if rapido else config.DISORDERS
    for disorder in disorders:
        for unit in config.UNITS:
            for analyzer in config.ANALYZERS:
                matrix = datasets.raw_tfidf(
                    disorder, unit, analyzer, control_svd)
                if rapido and unit == "message":
                    matrix = subsample_sparse(matrix, config.RAPIDO_MESSAGES)
                space = {"disorder": disorder, "unit": unit,
                         "analyzer": analyzer, "svd": NO_SVD}
                contrast = sparse_contrast(matrix, seed=config.RANDOM_STATE)
                for min_samples in datasets.min_samples_for(unit, rapido):
                    kd = sparse_kdist(matrix, min_samples)
                    rows.append(describe(
                        space, min_samples, kd, contrast, curves))

    config.RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(
        config.RESULTS_DIR / "e11_diagnostico.csv", index=False)
    pd.concat(curves).to_csv(
        config.RESULTS_DIR / "e11_kdist.csv", index=False)
