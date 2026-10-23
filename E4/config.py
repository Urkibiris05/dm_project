"""Configuración del experimento E4."""

from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = Path(__file__).resolve().parent / "resultados"
FIGURES_DIR: Path = Path(__file__).resolve().parent / "figuras"
DEFAULT_VECTORIZATION: str = "message__all__minilm"
P_VALUES: tuple[float, ...] = (0.5, 1.0, 2.0, float("inf"))
NORMALIZED_FLAGS: tuple[bool, ...] = (False, True)
MIN_SAMPLES_DEFAULT: int = 5
MIN_SAMPLES_VALUES: tuple[int, ...] = (3, 5, 8, 10, 15, 20)
MAX_SAMPLES_DEFAULT: int = 5000
EPS_QUANTILES: tuple[float, ...] = (0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95)
SEARCH_ROOTS: tuple[Path, ...] = (
    ROOT_DIR / "vectorizations",
    ROOT_DIR / "vectorizations" / "vectorizations",
    ROOT_DIR,
)

__all__ = [
    "ROOT_DIR",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "DEFAULT_VECTORIZATION",
    "P_VALUES",
    "NORMALIZED_FLAGS",
    "MIN_SAMPLES_DEFAULT",
    "MIN_SAMPLES_VALUES",
    "MAX_SAMPLES_DEFAULT",
    "EPS_QUANTILES",
    "SEARCH_ROOTS",
]
