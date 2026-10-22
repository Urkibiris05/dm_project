"""Experimento E4: análisis de DBSCAN con distancias Minkowski."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dbscan_implementation import DBSCAN
from evaluation.kdist import eps_grid, kdist
from evaluation.metrics import (
    external_metrics,
    internal_metrics,
    structure_summary,
)

from .config import (
    DEFAULT_VECTORIZATION,
    EPS_QUANTILES,
    MAX_SAMPLES_DEFAULT,
    MIN_SAMPLES_DEFAULT,
    P_VALUES,
    SEARCH_ROOTS,
)


def resolve_vectorization_dir(
    vectorization: str | Path | None = None,
) -> Path:
    """Resuelve la carpeta de la vectorización que se desea evaluar.

    Parameters
    ----------
    vectorization : str | Path | None
        Nombre de la carpeta, ruta relativa o ruta absoluta.

    Returns
    -------
    Path
        Ruta absoluta a la carpeta de la vectorización.

    Raises
    ------
    FileNotFoundError
        Si no existe ninguna carpeta compatible con la cadena indicada.
    """
    requested = (
        str(vectorization)
        if vectorization is not None
        else DEFAULT_VECTORIZATION
    )
    candidate = Path(requested).expanduser()
    if candidate.exists():
        return candidate.resolve()

    for base_path in SEARCH_ROOTS:
        possible = base_path / requested
        if possible.exists():
            return possible.resolve()

    raise FileNotFoundError(
        "No existe ninguna vectorización llamada "
        f"{requested!r}. Comprueba la carpeta vectorizations."
    )


def load_vectorization(
    vectorization: str | Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Carga la matriz de embbedings y la metadata asociada."""
    folder = resolve_vectorization_dir(vectorization)
    matrix_path = folder / "X.npy"
    metadata_path = folder / "metadata.csv"

    if not matrix_path.exists():
        raise FileNotFoundError(f"No existe la matriz X.npy en {folder}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"No existe metadata.csv en {folder}")

    points = np.load(matrix_path)
    metadata = pd.read_csv(metadata_path)
    if points.ndim != 2:
        raise ValueError(
            f"X.npy debe ser una matriz 2D; se recibió {points.shape}"
        )
    if len(points) != len(metadata):
        raise ValueError(
            "X.npy y metadata.csv tienen longitudes distintas: "
            f"{len(points)} vs {len(metadata)}"
        )
    return points, metadata


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normaliza cada fila del array con la norma L2."""
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe_norms


def _sample_rows(
    points: np.ndarray,
    metadata: pd.DataFrame,
    max_samples: int | None,
    seed: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Submuestrea filas para limitar el coste del experimento."""
    if max_samples is None or len(points) <= max_samples:
        return points, metadata

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_samples, replace=False)
    return points[indices], metadata.iloc[indices].reset_index(drop=True)


def _target_from_metadata(metadata: pd.DataFrame) -> np.ndarray:
    """Obtiene la etiqueta objetivo desde metadata.csv."""
    if "target" in metadata.columns:
        target = pd.to_numeric(metadata["target"], errors="coerce")
        if not target.isna().all():
            return target.fillna(0).to_numpy(dtype=int)
    if "label" in metadata.columns:
        target = pd.to_numeric(metadata["label"], errors="coerce")
        if not target.isna().all():
            return target.fillna(0).to_numpy(dtype=int)
    return np.zeros(len(metadata), dtype=int)


def _eps_candidates(
    points: np.ndarray,
    *,
    min_samples: int,
    p: float,
    normalized: bool,
) -> np.ndarray:
    """Genera eps con el k-dist del módulo evaluation."""
    working_points = _normalize_rows(points) if normalized else points
    kd_values = kdist(
        working_points,
        k=min_samples,
        metric="minkowski",
        p=p,
        block_size=1024,
    )
    return eps_grid(kd_values, EPS_QUANTILES)


def evaluate_vectorization(
    vectorization: str | Path,
    *,
    min_samples: int = MIN_SAMPLES_DEFAULT,
    max_samples: int = MAX_SAMPLES_DEFAULT,
    p_values: tuple[float, ...] = P_VALUES,
    normalized_flags: tuple[bool, ...] = (False, True),
    seed: int = 0,
) -> pd.DataFrame:
    """Evalúa una vectorización para cada combinación de p y normalización."""
    points, metadata = load_vectorization(vectorization)
    points, metadata = _sample_rows(points, metadata, max_samples, seed)
    target = _target_from_metadata(metadata)

    rows: list[dict[str, object]] = []
    for p_value in p_values:
        for normalized in normalized_flags:
            working_points = _normalize_rows(points) if normalized else points
            eps_values = _eps_candidates(
                points,
                min_samples=min_samples,
                p=p_value,
                normalized=normalized,
            )
            for eps_value in eps_values:
                model = DBSCAN(
                    eps=float(eps_value),
                    min_samples=min_samples,
                    metric="minkowski",
                    p=p_value,
                    algorithm="brute",
                )
                labels = model.fit_predict(working_points)
                summary = structure_summary(labels)
                metrics = internal_metrics(
                    working_points,
                    labels,
                    metric="minkowski",
                    sample_size=3000,
                    seed=seed,
                    p=p_value,
                )
                external = external_metrics(labels, target)
                rows.append(
                    {
                        "vectorization": str(vectorization),
                        "p": float(p_value),
                        "normalized": bool(normalized),
                        "eps": float(eps_value),
                        "min_samples": int(min_samples),
                        "n_clusters": int(summary["n_clusters"]),
                        "noise_rate": float(summary["noise_rate"]),
                        "largest_cluster_frac": float(
                            summary["largest_cluster_frac"]
                        ),
                        "dbcv": float(metrics.get("dbcv", float("nan"))),
                        "silhouette": float(
                            metrics.get("silhouette", float("nan"))
                        ),
                        "davies_bouldin": float(
                            metrics.get("davies_bouldin", float("nan"))
                        ),
                        "calinski_harabasz": float(
                            metrics.get("calinski_harabasz", float("nan"))
                        ),
                        "ari_all": float(
                            external.get("ari_all", float("nan"))
                        ),
                        "ami_all": float(
                            external.get("ami_all", float("nan"))
                        ),
                        "ari_nonoise": float(
                            external.get("ari_nonoise", float("nan"))
                        ),
                        "ami_nonoise": float(
                            external.get("ami_nonoise", float("nan"))
                        ),
                    }
                )

    return pd.DataFrame(rows)


def build_experiment_table(
    vectorization: str | Path,
    *,
    min_samples: int = MIN_SAMPLES_DEFAULT,
    max_samples: int = MAX_SAMPLES_DEFAULT,
    p_values: tuple[float, ...] = P_VALUES,
    normalized_flags: tuple[bool, ...] = (False, True),
    seed: int = 0,
) -> pd.DataFrame:
    """Construye la tabla completa de resultados del experimento E4."""
    table = evaluate_vectorization(
        vectorization,
        min_samples=min_samples,
        max_samples=max_samples,
        p_values=p_values,
        normalized_flags=normalized_flags,
        seed=seed,
    )
    if table.empty:
        return table
    return table.sort_values(
        ["p", "normalized", "eps"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


__all__ = [
    "resolve_vectorization_dir",
    "load_vectorization",
    "evaluate_vectorization",
    "build_experiment_table",
]
