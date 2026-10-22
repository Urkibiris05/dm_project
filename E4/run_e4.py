"""Ejecuta E4 sobre una vectorización y guarda métricas en CSV.

Uso:
    python -m E4.run_e4 --vectorization /ruta/a/vectorizacion
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from evaluation.metrics import (
    external_metrics,
    internal_metrics,
    structure_summary,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "resultados"
DEFAULT_MIN_SAMPLES = (5, 10, 20)
DEFAULT_PERCENTILES = (10, 20, 30, 40, 50, 60, 70, 80, 90)


def parse_int_list(raw: str) -> tuple[int, ...]:
    """Convierte una lista separada por comas a una tupla de enteros."""
    values = tuple(
        int(value.strip()) for value in raw.split(",") if value.strip()
    )
    if not values:
        raise ValueError("Se requiere al menos un valor entero.")
    return values


def resolve_vectorization_paths(vectorization: Path) -> tuple[Path, Path]:
    """Devuelve rutas de ``X.npy`` y ``metadata.csv`` desde archivo o carpeta.
    """
    if vectorization.is_dir():
        x_path = vectorization / "X.npy"
        metadata_path = vectorization / "metadata.csv"
    else:
        x_path = vectorization
        metadata_path = vectorization.parent / "metadata.csv"
    if not x_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "La vectorización debe incluir X.npy y metadata.csv"
        )
    return x_path, metadata_path


def load_vectorization(vectorization: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Carga la matriz y metadatos de una vectorización."""
    x_path, metadata_path = resolve_vectorization_paths(vectorization)
    return np.load(x_path), pd.read_csv(metadata_path)


def evaluate_configuration(
    X: np.ndarray,
    metadata: pd.DataFrame,
    eps: float,
    min_samples: int,
    metric: str,
) -> dict[str, float | int]:
    """Ajusta DBSCAN para una configuración y calcula métricas de evaluación.
    """
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        algorithm="brute",
        n_jobs=-1,
    ).fit_predict(X)

    row: dict[str, float | int] = {
        "eps": eps,
        "min_samples": min_samples,
        **structure_summary(labels),
        **internal_metrics(X, labels, metric=metric),
    }

    for target_column in ("label", "disorder"):
        if target_column not in metadata:
            continue
        target = (
            metadata[target_column].astype("category").cat.codes.to_numpy()
        )
        scores = external_metrics(labels, target)
        for name, value in scores.items():
            row[f"{target_column}_{name}"] = value
    return row


def run_experiment(
    vectorization: Path,
    metric: str,
    min_samples_values: tuple[int, ...],
    percentiles: tuple[int, ...],
) -> pd.DataFrame:
    """Ejecuta el barrido de E4 y devuelve el grid completo."""
    X, metadata = load_vectorization(vectorization)
    rows = []
    for min_samples in min_samples_values:
        distances, _ = NearestNeighbors(
            n_neighbors=min_samples,
            metric=metric,
            algorithm="brute",
            n_jobs=-1,
        ).fit(X).kneighbors(X)
        kdist = np.sort(distances[:, -1])
        for percentile in percentiles:
            eps = float(np.percentile(kdist, percentile))
            row = evaluate_configuration(X, metadata, eps, min_samples, metric)
            row["percentile"] = percentile
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["min_samples", "percentile"])


def select_best_configuration(results: pd.DataFrame) -> pd.DataFrame:
    """Selecciona la mejor fila priorizando DBCV y menor ruido."""
    ranked = results.copy()
    ranked["dbcv_rank"] = ranked["dbcv"].fillna(-np.inf)
    return ranked.sort_values(
        ["dbcv_rank", "noise_rate", "largest_cluster_frac"],
        ascending=[False, True, True],
    ).head(1).drop(columns=["dbcv_rank"])


def save_results(results: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Guarda el grid y la mejor configuración en archivos CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "e4_grid.csv"
    best_path = output_dir / "e4_best.csv"
    results.to_csv(grid_path, index=False)
    select_best_configuration(results).to_csv(best_path, index=False)
    return grid_path, best_path


def main() -> None:
    """Punto de entrada CLI para ejecutar E4."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vectorization",
        type=Path,
        required=True,
        help="Carpeta de vectorización (o ruta a X.npy).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta donde se guardan los CSV de resultados.",
    )
    parser.add_argument(
        "--metric",
        choices=("cosine", "euclidean"),
        default="cosine",
        help="Métrica usada en NearestNeighbors y DBSCAN.",
    )
    parser.add_argument(
        "--min-samples",
        default=",".join(str(value) for value in DEFAULT_MIN_SAMPLES),
        help="Valores de min_samples separados por coma, p.ej. 5,10,20.",
    )
    parser.add_argument(
        "--percentiles",
        default=",".join(str(value) for value in DEFAULT_PERCENTILES),
        help="Percentiles de k-dist separados por coma, p.ej. 10,20,30.",
    )
    args = parser.parse_args()

    min_samples_values = parse_int_list(args.min_samples)
    percentiles = parse_int_list(args.percentiles)

    results = run_experiment(
        vectorization=args.vectorization,
        metric=args.metric,
        min_samples_values=min_samples_values,
        percentiles=percentiles,
    )
    grid_path, best_path = save_results(results, args.output_dir)
    print(f"Grid completo guardado en {grid_path}")
    print(f"Mejor configuración guardada en {best_path}")


if __name__ == "__main__":
    main()
