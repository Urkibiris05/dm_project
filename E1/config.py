"""Constantes del experimento E1 (TF-IDF por palabra y por caracter)."""

from pathlib import Path

import numpy as np

RANDOM_STATE: int = 170

DISORDERS: tuple[str, ...] = ("Ansiedad", "Depresion", "TCA")
ANALYZERS: tuple[str, ...] = ("word", "char_wb")
UNITS: tuple[str, ...] = ("message", "user")
SVD_DIMS: tuple[int, ...] = (32, 64, 128, 300)
NGRAMS: dict[str, tuple[int, int]] = {"word": (1, 1), "char_wb": (1, 5)}

# Los vectores salen normalizados (L2), así que coseno y euclídea dan la
# misma partición (E0.2): se usa coseno y no se compara con euclídea.
METRIC: str = "cosine"

# minPts según la unidad: más instancias, más densidad mínima.
MIN_SAMPLES_GRID: dict[str, tuple[int, ...]] = {
    "message": (5, 10, 20, 40),
    "user": (3, 5, 10),
}
# Cuantiles del k-dist que se usan como eps (se añade el codo).
EPS_QUANTILES: np.ndarray = np.linspace(0.1, 0.9, 9)

N_BOOTSTRAP: int = 20
DBCV_MAX_POINTS: int = 3000
KDIST_POINTS: int = 500  # puntos de cada curva k-dist que se guardan

# Modo --rapido: un solo espacio pequeño para depurar.
RAPIDO_DISORDERS: tuple[str, ...] = ("TCA",)
RAPIDO_SVD: tuple[int, ...] = (64,)
RAPIDO_MESSAGES: int = 2000
RAPIDO_QUANTILES: np.ndarray = np.array([0.3, 0.5, 0.7])
RAPIDO_MIN_SAMPLES: dict[str, tuple[int, ...]] = {
    "message": (5, 10), "user": (3, 5)}
RAPIDO_BOOTSTRAP: int = 3

# Columnas que identifican un espacio de vectorización.
SPACE_COLUMNS: list[str] = ["disorder", "unit", "analyzer", "svd"]

ROOT_DIR: Path = Path(__file__).parent.parent
VECTORIZATIONS_DIR: Path = ROOT_DIR / "vectorizations"
E1_DIR: Path = Path(__file__).parent
RESULTS_DIR: Path = E1_DIR / "resultados"
FIGURES_DIR: Path = E1_DIR / "figuras"
