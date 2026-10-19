"""E1.3: estabilidad de las mejores configuraciones y selección final.

Solo a las 5 mejores válidas por DBCV de cada espacio se les calcula la
estabilidad (bootstrap y perturbación de eps). Después se aplica la regla de
``evaluation.selection``. Nada de esto usa etiquetas Gold.
"""

import pandas as pd

from evaluation.selection import select, shortlist
from evaluation.stability import bootstrap_stability, eps_perturbation

from . import config, datasets


def main(rapido: bool = False) -> None:
    grid = pd.read_csv(config.RESULTS_DIR / "e12_grid.csv")
    n_runs = config.RAPIDO_BOOTSTRAP if rapido else config.N_BOOTSTRAP
    top = shortlist(grid, config.SPACE_COLUMNS)
    grid["stab_mean"] = float("nan")
    grid["stab_std"] = float("nan")
    grid["stab_p10"] = float("nan")
    grid["eps_ari"] = float("nan")

    cache: dict[tuple, object] = {}
    for index in top:
        row = grid.loc[index]
        space = {key: row[key] for key in config.SPACE_COLUMNS}
        key = tuple(space.values())
        if key not in cache:
            cache.clear()
            space["svd"] = int(space["svd"])
            cache[key] = datasets.load_for(space, rapido)[0]
        X = cache[key]
        eps, min_samples = float(row["eps"]), int(row["min_samples"])
        result = bootstrap_stability(
            X, eps, min_samples, config.METRIC, n_runs=n_runs,
            seed=config.RANDOM_STATE)
        for name, value in result.items():
            grid.loc[index, name] = value
        grid.loc[index, "eps_ari"] = eps_perturbation(
            X, eps, min_samples, config.METRIC)

    grid.to_csv(config.RESULTS_DIR / "e13_estabilidad.csv", index=False)
    chosen = select(grid, config.SPACE_COLUMNS)
    chosen.to_csv(config.RESULTS_DIR / "e13_seleccion.csv", index=False)

    # Robustez del espacio: fracción de la rejilla que es válida.
    robustness = (grid.groupby(config.SPACE_COLUMNS)["is_valid"]
                  .mean().rename("pct_valid").reset_index())
    robustness.to_csv(
        config.RESULTS_DIR / "e13_robustez_espacio.csv", index=False)
