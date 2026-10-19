"""Carga de las vectorizaciones TF-IDF de E1 (una carpeta por espacio)."""

from collections.abc import Iterator
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import Normalizer

from . import config


def space_name(disorder: str, unit: str, analyzer: str, svd: int) -> str:
    """Nombre de la carpeta, igual que ``vectorization/naming.py``."""
    low, high = config.NGRAMS[analyzer]
    return f"{unit}__{disorder}__tfidf-{analyzer}-{low}-{high}-svd{svd}"


def space_dir(disorder: str, unit: str, analyzer: str, svd: int) -> Path:
    """Carpeta de la vectorización; falla con un mensaje claro si falta."""
    path = config.VECTORIZATIONS_DIR / space_name(
        disorder, unit, analyzer, svd)
    if not path.exists():
        raise FileNotFoundError(
            f"Falta {path}: ejecuta run_vectorizations.sh (líneas de E1)")
    return path


def load_space(
    disorder: str, unit: str, analyzer: str, svd: int
) -> tuple[np.ndarray, pd.DataFrame]:
    """Devuelve ``(X, metadata)``: la matriz (float32) y sus metadatos."""
    path = space_dir(disorder, unit, analyzer, svd)
    X = np.load(path / "X.npy")
    metadata = pd.read_csv(path / "metadata.csv")
    return X, metadata


def raw_tfidf(
    disorder: str, unit: str, analyzer: str, svd: int
) -> sparse.csr_matrix:
    """TF-IDF normalizado SIN la SVD (control del "± SVD").

    Reutiliza el pipeline ajustado (``tfidf -> svd -> normalizer``) y aplica
    solo sus dos pasos extremos a los textos de ``metadata.csv``. El
    vocabulario es el mismo que el del espacio con SVD.
    """
    path = space_dir(disorder, unit, analyzer, svd)
    pipeline = joblib.load(path / "vectorizer.joblib")
    metadata = pd.read_csv(path / "metadata.csv")
    matrix = pipeline.named_steps["tfidf"].transform(
        metadata["text"].fillna("").tolist())
    return Normalizer(norm="l2").fit_transform(matrix).tocsr()


def subsample(
    X: np.ndarray, metadata: pd.DataFrame, n_rows: int, seed: int
) -> tuple[np.ndarray, pd.DataFrame]:
    """Submuestra aleatoria reproducible (solo para el modo rápido)."""
    if len(X) <= n_rows:
        return X, metadata
    rows = np.sort(np.random.default_rng(seed).choice(
        len(X), size=n_rows, replace=False))
    return X[rows], metadata.iloc[rows].reset_index(drop=True)


def iter_spaces(rapido: bool = False) -> Iterator[dict]:
    """Recorre los espacios de E1 como diccionarios con sus cuatro claves."""
    disorders = config.RAPIDO_DISORDERS if rapido else config.DISORDERS
    dims = config.RAPIDO_SVD if rapido else config.SVD_DIMS
    for disorder in disorders:
        for unit in config.UNITS:
            for analyzer in config.ANALYZERS:
                for svd in dims:
                    yield {"disorder": disorder, "unit": unit,
                           "analyzer": analyzer, "svd": svd}


def load_for(
    space: dict, rapido: bool = False
) -> tuple[np.ndarray, pd.DataFrame]:
    """Carga un espacio (``iter_spaces``); en modo rápido lo submuestrea."""
    X, metadata = load_space(**space)
    if rapido and space["unit"] == "message":
        X, metadata = subsample(
            X, metadata, config.RAPIDO_MESSAGES, config.RANDOM_STATE)
    return X, metadata


def min_samples_for(unit: str, rapido: bool = False) -> tuple[int, ...]:
    """Rejilla de minPts para una unidad de análisis."""
    grid = config.RAPIDO_MIN_SAMPLES if rapido else config.MIN_SAMPLES_GRID
    return grid[unit]


def quantiles_for(rapido: bool = False) -> np.ndarray:
    """Cuantiles del k-dist que se usan como eps."""
    return config.RAPIDO_QUANTILES if rapido else config.EPS_QUANTILES
