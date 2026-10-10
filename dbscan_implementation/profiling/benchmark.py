"""Compara tiempo y memoria de varias implementaciones de DBSCAN.

Uso (desde la raíz del repositorio, en WSL)::

    python -m dbscan_implementation.profiling.benchmark
    python -m dbscan_implementation.profiling.plots

El primer comando escribe ``profiling/results/results.csv`` y el segundo
dibuja las gráficas a partir de ese archivo.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, pairwise_distances

from .implementations import IMPLEMENTATIONS
from .measure import make_data

RESULTS_DIR: Path = Path(__file__).parent / "results"
RESULTS_FILE: Path = RESULTS_DIR / "results.csv"

# Implementación con la que se comparan las etiquetas de las demás.
REFERENCE: str = "sklearn_brute"

MIN_SAMPLES: int = 5
TIMEOUT_SECONDS: int = 900

CSV_COLUMNS: list[str] = [
    "scenario", "metric", "n_features", "n_samples", "eps", "min_samples",
    "implementation", "seconds", "memory_mb", "n_clusters", "noise_rate",
    "ari_vs_sklearn",
]


@dataclass
class Scenario:
    """Un tipo de datos sobre el que se mide."""

    name: str
    metric: str
    n_features: int


SCENARIOS: list[Scenario] = [
    Scenario("euclidea_2d", "euclidean", 2),
    Scenario("euclidea_32d", "euclidean", 32),
    Scenario("coseno_32d", "cosine", 32),
]


def choose_eps(scenario: Scenario) -> float:
    """Elige ``eps`` para que cada punto tenga ~1 % de los datos de vecinos.

    Es el percentil 1 de las distancias entre los puntos de una muestra.
    Así el radio es razonable en todos los escenarios sin ajustarlo a mano.
    """
    sample = make_data(500, scenario.n_features)
    distances = pairwise_distances(sample, metric=scenario.metric)
    upper_triangle = distances[np.triu_indices_from(distances, k=1)]
    return float(np.percentile(upper_triangle, 1))


def run_case(
    implementation_name: str,
    scenario: Scenario,
    n_samples: int,
    eps: float,
    repeats: int,
    threads: int | None,
) -> tuple[dict, np.ndarray] | None:
    """Mide un caso en un proceso aparte. Devuelve ``None`` si falla."""
    environment = os.environ.copy()
    if threads is not None:
        for variable in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "PARLAY_NUM_THREADS",
        ):
            environment[variable] = str(threads)

    with tempfile.TemporaryDirectory() as folder:
        labels_file = os.path.join(folder, "labels.npy")
        command = [
            sys.executable, "-m", "dbscan_implementation.profiling.measure",
            "--implementation", implementation_name,
            "--metric", scenario.metric,
            "--n-samples", str(n_samples),
            "--n-features", str(scenario.n_features),
            "--eps", repr(eps),
            "--min-samples", str(MIN_SAMPLES),
            "--repeats", str(repeats),
            "--labels-file", labels_file,
        ]
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, env=environment,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            print("    (tiempo límite superado)")
            return None
        if process.returncode != 0:
            print("    (error)", process.stderr.strip().splitlines()[-1:])
            return None

        measurement = json.loads(process.stdout.strip().splitlines()[-1])
        labels = np.load(labels_file)
    return measurement, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[500, 1000, 2000, 4000, 8000, 16000],
        help="números de puntos que se prueban",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="repeticiones por caso (se guarda la mediana del tiempo)",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="limita los hilos de las librerías (por defecto, sin límite)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    rows: list[dict] = []

    for scenario in SCENARIOS:
        eps = choose_eps(scenario)
        print(f"== {scenario.name} (eps={eps:.4g})")
        # Si una implementación falla con un tamaño, no se prueban mayores.
        failed: set[str] = set()

        for n_samples in args.sizes:
            results: dict[str, tuple[dict, np.ndarray]] = {}
            for implementation in IMPLEMENTATIONS:
                name = implementation.name
                if scenario.metric not in implementation.metrics:
                    continue
                max_features = implementation.max_features
                if max_features and scenario.n_features > max_features:
                    continue
                if name in failed:
                    continue
                print(f"  n={n_samples:>6}  {name}")
                result = run_case(
                    name, scenario, n_samples, eps, args.repeats,
                    args.threads,
                )
                if result is None:
                    failed.add(name)
                else:
                    results[name] = result

            for name, (measurement, labels) in results.items():
                if REFERENCE in results:
                    ari = adjusted_rand_score(results[REFERENCE][1], labels)
                else:
                    ari = float("nan")
                rows.append({
                    "scenario": scenario.name,
                    "metric": scenario.metric,
                    "n_features": scenario.n_features,
                    "n_samples": n_samples,
                    "eps": eps,
                    "min_samples": MIN_SAMPLES,
                    "implementation": name,
                    "seconds": measurement["seconds"],
                    "memory_mb": measurement["memory_mb"],
                    "n_clusters": len(set(labels.tolist()) - {-1}),
                    "noise_rate": float(np.mean(labels == -1)),
                    "ari_vs_sklearn": ari,
                })

    with open(RESULTS_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Resultados guardados en {RESULTS_FILE}")


if __name__ == "__main__":
    main()
