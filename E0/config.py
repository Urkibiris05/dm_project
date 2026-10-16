"""Constantes del experimento E0 (corrección de nuestro DBSCAN)."""

from pathlib import Path

import numpy as np

RANDOM_STATE: int = 170
N_SAMPLES: int = 1000  # puntos de los datasets sintéticos 2D

MIN_SAMPLES_GRID: tuple[int, ...] = (3, 5, 10)

# Cuantiles del k-dist que se usan como valores de eps. El cuantil q hace que
# aproximadamente una fracción q de los puntos sea núcleo.
EPS_QUANTILES: np.ndarray = np.linspace(0.1, 0.9, 9)
E03_QUANTILES: tuple[float, ...] = (0.3, 0.5, 0.7)

# (nombre de la métrica, p). p solo se usa con "minkowski".
METRICS: list[tuple[str, float | None]] = [
    ("euclidean", None),
    ("manhattan", None),
    ("chebyshev", None),
    ("minkowski", 3),
    ("cosine", None),
]

# Variantes de nuestro DBSCAN: nombre -> argumentos extra de DBSCAN.
OUR_VARIANTS: dict[str, dict] = {
    "ours_brute": {"algorithm": "brute"},
    "ours_blocked": {"algorithm": "blocked"},
    "ours_optimized_blocked": {"algorithm": "optimized_blocked"},
    "ours_optimized_dedup": {
        "algorithm": "optimized_blocked",
        "deduplicate_queue": True,
    },
}

N_PERMUTATIONS: int = 20
N_REPEATS: int = 5

# Tolerancia relativa para decir que una distancia "empata" con eps.
BOUNDARY_TOL: float = 1e-9

E0_DIR: Path = Path(__file__).parent
RESULTS_DIR: Path = E0_DIR / "resultados"
FIGURES_DIR: Path = E0_DIR / "figuras"
