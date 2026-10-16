"""Ejecuta todo E0: python -m E0.run_e0 [--rapido] (desde la raíz, en WSL)."""

import argparse
import time

import pandas as pd

from . import (
    exp_cos_vs_l2, exp_libraries, exp_vs_sklearn, plots, verify_ties,
)
from .config import FIGURES_DIR, RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rapido", action="store_true",
        help="rejillas reducidas para depurar",
    )
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    steps = [
        ("E0.1 nuestro DBSCAN vs sklearn", exp_vs_sklearn.main),
        ("E0.2 coseno vs euclidea normalizada", exp_cos_vs_l2.main),
        ("E0.3 otras librerias", exp_libraries.main),
        ("Verificacion de empates", verify_ties.main),
    ]
    for title, function in steps:
        print(f"== {title}")
        start = time.perf_counter()
        function(args.rapido)
        print(f"   {time.perf_counter() - start:.1f} s")

    print("== Figuras")
    plots.main()

    print("== Resumen")
    pd.set_option("display.width", 120)
    summary = pd.read_csv(RESULTS_DIR / "e01_resumen.csv")
    print(summary.groupby("variant")["pct_exact_equal"].min())
    print(pd.read_csv(RESULTS_DIR / "e02_resumen.csv"))
    print(pd.read_csv(RESULTS_DIR / "e03_reproducibilidad_resumen.csv"))


if __name__ == "__main__":
    main()
