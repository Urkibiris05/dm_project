"""E0.3: concordancia y reproducibilidad frente a otras librerías.

Reutiliza los runners de dbscan_implementation/profiling/implementations.py
(nuestras variantes, scikit-learn, dbscan de PyPI y mlpack).
"""

import itertools

import numpy as np
import pandas as pd

from dbscan_implementation.profiling.implementations import IMPLEMENTATIONS

from . import analysis
from .config import (
    E03_QUANTILES,
    N_PERMUTATIONS,
    N_REPEATS,
    RESULTS_DIR,
)
from .datasets import distance_matrix, eps_grid, load_datasets

DATASET_NAMES: tuple[str, ...] = (
    "blobs", "varied", "aniso", "moons", "circles", "no_structure",
    "iris", "wine", "digits",
)
MIN_SAMPLES: int = 5
REFERENCE: str = "sklearn_brute"
REPRODUCIBILITY_QUANTILE: float = 0.5


def usable_implementations(n_features: int) -> list:
    """Implementaciones euclídeas que admiten esa dimensionalidad."""
    return [
        impl for impl in IMPLEMENTATIONS
        if "euclidean" in impl.metrics
        and not (impl.max_features and n_features > impl.max_features)
    ]


def run_all(X: np.ndarray, eps: float) -> tuple[dict, dict]:
    """Ejecuta cada implementación. Devuelve (etiquetas, errores)."""
    labels: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    for impl in usable_implementations(X.shape[1]):
        try:
            labels[impl.name] = np.asarray(
                impl.run(X.copy(), eps, MIN_SAMPLES, "euclidean")
            )
        except Exception as error:  # una librería rota no aborta E0
            errors[impl.name] = repr(error)
    return labels, errors


def agreement_and_disagreements(
    datasets: dict, quantiles: tuple[float, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ARI entre pares y desglose de desacuerdos frente a sklearn."""
    pair_rows: list[dict] = []
    diff_rows: list[dict] = []
    for name in DATASET_NAMES:
        X = datasets[name].X
        D = distance_matrix(X, "euclidean")
        for quantile, eps in zip(
            quantiles, eps_grid(D, MIN_SAMPLES, quantiles)
        ):
            eps = float(eps)
            labels, errors = run_all(X, eps)
            for impl_name, message in errors.items():
                print(f"    aviso: {impl_name} falló en {name}: {message}")
            neigh = analysis.neighborhoods_reference(X, eps, "euclidean")
            core = analysis.core_mask(neigh, MIN_SAMPLES)
            for a, b in itertools.combinations(labels, 2):
                pair_rows.append({
                    "dataset": name, "eps_quantile": quantile,
                    "impl_a": a, "impl_b": b,
                    "ari": analysis.compare_partitions(
                        labels[a], labels[b], core)["ari"],
                })
            for impl_name, impl_labels in labels.items():
                row = {"dataset": name, "eps_quantile": quantile,
                       "implementation": impl_name}
                row.update(analysis.compare_partitions(
                    labels[REFERENCE], impl_labels, core))
                row.update(analysis.dbscan_validity(
                    impl_labels, neigh, core))
                diff_rows.append(row)
        print(f"  E0.3 concordancia {name} lista")
    return pd.DataFrame(pair_rows), pd.DataFrame(diff_rows)


def _row(
    dataset: str, implementation: str, kind: str, seed: int,
    base: np.ndarray, other: np.ndarray, core: np.ndarray,
) -> dict:
    comparison = analysis.compare_partitions(base, other, core)
    return {
        "dataset": dataset, "implementation": implementation, "kind": kind,
        "seed": seed, "exact_equal": comparison["exact_equal"],
        "ari": comparison["ari"], "same_noise": comparison["same_noise"],
        "core_ari": comparison["core_ari"],
        "n_diff_border": comparison["n_diff_border"],
    }


def reproducibility(
    datasets: dict, n_repeats: int, n_permutations: int
) -> pd.DataFrame:
    """Determinismo (misma entrada) e invariancia al orden (permutaciones).

    Cada implementación se compara consigo misma: las etiquetas de una
    ejecución con la entrada permutada se devuelven al orden original.
    """
    rows: list[dict] = []
    for name in DATASET_NAMES:
        X = datasets[name].X
        D = distance_matrix(X, "euclidean")
        eps = float(eps_grid(D, MIN_SAMPLES, (REPRODUCIBILITY_QUANTILE,))[0])
        neigh = analysis.neighborhoods_reference(X, eps, "euclidean")
        core = analysis.core_mask(neigh, MIN_SAMPLES)
        for impl in usable_implementations(X.shape[1]):
            try:
                base = np.asarray(
                    impl.run(X.copy(), eps, MIN_SAMPLES, "euclidean")
                )
                for seed in range(n_repeats):
                    again = np.asarray(
                        impl.run(X.copy(), eps, MIN_SAMPLES, "euclidean")
                    )
                    rows.append(_row(name, impl.name, "repeat", seed,
                                     base, again, core))
                for seed in range(n_permutations):
                    perm = np.random.default_rng(seed).permutation(len(X))
                    permuted = np.asarray(impl.run(
                        X[perm].copy(), eps, MIN_SAMPLES, "euclidean"
                    ))
                    back = np.empty_like(permuted)
                    back[perm] = permuted
                    rows.append(_row(name, impl.name, "permutation", seed,
                                     base, back, core))
            except Exception as error:
                print(f"    aviso: {impl.name} falló en {name}: {error!r}")
        print(f"  E0.3 reproducibilidad {name} lista")
    return pd.DataFrame(rows)


def reproducibility_summary(repro: pd.DataFrame) -> pd.DataFrame:
    """Resumen por implementación de la tabla de reproducibilidad."""
    rows = []
    for implementation, group in repro.groupby("implementation"):
        repeats = group[group["kind"] == "repeat"]
        perms = group[group["kind"] == "permutation"]
        rows.append({
            "implementation": implementation,
            "determinista": bool(repeats["exact_equal"].all()),
            "ari_perm_media": perms["ari"].mean(),
            "ari_perm_min": perms["ari"].min(),
            "pct_same_noise": 100 * perms["same_noise"].mean(),
            "core_ari_min": perms["core_ari"].min(),
        })
    return pd.DataFrame(rows)


def main(rapido: bool = False) -> None:
    datasets = load_datasets()
    quantiles = (0.5,) if rapido else E03_QUANTILES
    n_repeats = 2 if rapido else N_REPEATS
    n_permutations = 3 if rapido else N_PERMUTATIONS

    pairs, differences = agreement_and_disagreements(datasets, quantiles)
    pairs.to_csv(RESULTS_DIR / "e03_ari_pares.csv", index=False)
    differences.to_csv(RESULTS_DIR / "e03_desacuerdos.csv", index=False)

    # Matriz simétrica con el ARI medio entre cada par de implementaciones.
    mean_ari = pairs.groupby(["impl_a", "impl_b"])["ari"].mean()
    present = set(pairs["impl_a"]) | set(pairs["impl_b"])
    names = [i.name for i in IMPLEMENTATIONS if i.name in present]
    matrix = pd.DataFrame(1.0, index=names, columns=names)
    for (a, b), value in mean_ari.items():
        matrix.loc[a, b] = value
        matrix.loc[b, a] = value
    matrix.to_csv(RESULTS_DIR / "e03_ari_matriz.csv")

    repro = reproducibility(datasets, n_repeats, n_permutations)
    repro.to_csv(RESULTS_DIR / "e03_reproducibilidad.csv", index=False)
    reproducibility_summary(repro).to_csv(
        RESULTS_DIR / "e03_reproducibilidad_resumen.csv", index=False
    )
