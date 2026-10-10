"""Mide una sola ejecución de DBSCAN en un proceso limpio.

``benchmark.py`` lanza este archivo una vez por cada caso. Usar un proceso
nuevo para cada medida evita que la memoria de un caso contamine al
siguiente. No está pensado para ejecutarse a mano.

Solo funciona en Linux (o WSL): la memoria se lee de ``/proc``.
"""

import argparse
import json
import statistics
import time

import numpy as np
from sklearn.datasets import make_blobs

from .implementations import get_implementation

WARM_UP_POINTS: int = 50


def make_data(n_samples: int, n_features: int) -> np.ndarray:
    """Datos sintéticos: cinco grupos gaussianos."""
    X, _ = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=5, random_state=0
    )
    return X


def read_memory_kb(field: str) -> int:
    """Lee un campo de ``/proc/self/status`` (en kB).

    ``VmRSS`` es la memoria que el proceso ocupa ahora mismo y ``VmHWM``
    es el máximo que ha llegado a ocupar.
    """
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    raise RuntimeError(f"No se encontró {field} en /proc/self/status")


def reset_peak_memory() -> None:
    """Pone a cero el máximo de memoria (``VmHWM``) del proceso."""
    with open("/proc/self/clear_refs", "w") as clear_refs:
        clear_refs.write("5")


def measure(
    implementation_name: str,
    metric: str,
    n_samples: int,
    n_features: int,
    eps: float,
    min_samples: int,
    repeats: int,
) -> tuple[float, float, np.ndarray]:
    """Devuelve (segundos, MB de memoria extra en el pico, etiquetas)."""
    implementation = get_implementation(implementation_name)
    X = make_data(n_samples, n_features)

    # Una ejecución pequeña antes de medir, para que la carga de librerías
    # y la creación de hilos no cuenten como tiempo ni memoria del algoritmo.
    implementation.run(X[:WARM_UP_POINTS], eps, min_samples, metric)

    reset_peak_memory()
    memory_before_kb = read_memory_kb("VmRSS")

    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        labels = implementation.run(X, eps, min_samples, metric)
        seconds.append(time.perf_counter() - start)

    peak_kb = read_memory_kb("VmHWM")
    extra_mb = max(peak_kb - memory_before_kb, 0) / 1024
    return statistics.median(seconds), extra_mb, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--n-features", type=int, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--min-samples", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--labels-file", required=True)
    args = parser.parse_args()

    seconds, memory_mb, labels = measure(
        args.implementation,
        args.metric,
        args.n_samples,
        args.n_features,
        args.eps,
        args.min_samples,
        args.repeats,
    )
    np.save(args.labels_file, labels)
    print(json.dumps({"seconds": seconds, "memory_mb": memory_mb}))


if __name__ == "__main__":
    main()
