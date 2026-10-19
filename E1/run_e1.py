"""Ejecuta todo E1: python -m E1.run_e1 [--rapido] (desde la raíz, en WSL).

Antes hay que generar las vectorizaciones: bash run_vectorizations.sh
"""

import argparse
import time

import pandas as pd

from . import (exp_grid, exp_interpret, exp_kdist, exp_stability, plots)
from .config import FIGURES_DIR, RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rapido", action="store_true",
        help="un solo trastorno, SVD 64 y rejillas reducidas para depurar",
    )
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    steps = [
        ("E1.1 diagnostico de densidad (k-dist)", exp_kdist.main),
        ("E1.2 barrido de eps y minPts", exp_grid.main),
        ("E1.3 estabilidad y seleccion", exp_stability.main),
        ("E1.4 evaluacion externa e interpretacion", exp_interpret.main),
    ]
    for title, function in steps:
        print(f"== {title}")
        start = time.perf_counter()
        function(args.rapido)
        print(f"   {time.perf_counter() - start:.1f} s")

    print("== Figuras")
    plots.main()

    print("== Resumen")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    chosen = pd.read_csv(RESULTS_DIR / "e13_seleccion.csv")
    columns = ["disorder", "unit", "analyzer", "svd", "min_samples", "eps",
               "n_clusters", "noise_rate", "largest_cluster_frac", "dbcv",
               "stab_p10", "status"]
    print(chosen[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
