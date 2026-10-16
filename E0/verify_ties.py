"""Verifica que las discrepancias de E0.1 y E0.2 son empates numéricos.

Si una partición solo difiere porque algún par de puntos está a distancia
exactamente eps (y cada implementación lo redondea distinto), al desplazar
eps una cantidad ínfima (1e-7 relativo) las dos implementaciones deben
volver a coincidir.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN as SklearnDBSCAN

from dbscan_implementation import DBSCAN

from .config import OUR_VARIANTS, RESULTS_DIR
from .datasets import load_datasets, normalize_rows

SHIFTS: tuple[float, ...] = (-1e-7, 1e-7)


def verify_e01(datasets: dict) -> list[dict]:
    """Repite cada discrepancia de E0.1 con eps desplazado."""
    path = RESULTS_DIR / "e01_discrepancias.csv"
    table = pd.read_csv(path)
    rows: list[dict] = []
    for _, case in table.iterrows():
        X = datasets[case["dataset"]].X
        p = None if pd.isna(case["p"]) else float(case["p"])
        kwargs = {"p": p} if case["metric"] == "minkowski" else {}
        for shift in SHIFTS:
            eps = float(case["eps"]) * (1 + shift)
            reference = SklearnDBSCAN(
                eps=eps, min_samples=int(case["min_samples"]),
                metric=case["metric"], algorithm="brute", **kwargs,
            ).fit(X)
            ours = DBSCAN(
                eps=eps, min_samples=int(case["min_samples"]),
                metric=case["metric"], p=p,
                **OUR_VARIANTS[case["variant"]],
            ).fit(X)
            rows.append({
                "experimento": "E0.1",
                "dataset": case["dataset"],
                "metric": case["metric"],
                "min_samples": int(case["min_samples"]),
                "eps": case["eps"],
                "variante": case["variant"],
                "desplazamiento": shift,
                "exact_equal": bool(
                    np.array_equal(reference.labels_, ours.labels_)
                ),
            })
    return rows


def verify_e02(datasets: dict) -> list[dict]:
    """Repite cada discrepancia coseno vs euclídea de E0.2."""
    table = pd.read_csv(RESULTS_DIR / "e02_cos_vs_l2.csv")
    bad = table[table["normalized"] & ~table["exact_equal"]]
    rows: list[dict] = []
    for _, case in bad.iterrows():
        X = normalize_rows(datasets[case["dataset"]].X)
        for shift in SHIFTS:
            eps_cos = float(case["eps_cos"]) * (1 + shift)
            eps_l2 = float(np.sqrt(2 * eps_cos))
            labels = []
            for metric, eps in (("cosine", eps_cos), ("euclidean", eps_l2)):
                if case["implementation"] == "sklearn_brute":
                    model = SklearnDBSCAN(
                        eps=eps, min_samples=int(case["min_samples"]),
                        metric=metric, algorithm="brute",
                    )
                else:
                    model = DBSCAN(
                        eps=eps, min_samples=int(case["min_samples"]),
                        metric=metric,
                        **OUR_VARIANTS[case["implementation"]],
                    )
                labels.append(model.fit(X).labels_)
            rows.append({
                "experimento": "E0.2",
                "dataset": case["dataset"],
                "metric": "cosine vs euclidean",
                "min_samples": int(case["min_samples"]),
                "eps": case["eps_cos"],
                "variante": case["implementation"],
                "desplazamiento": shift,
                "exact_equal": bool(np.array_equal(*labels)),
            })
    return rows


def main(rapido: bool = False) -> None:
    datasets = load_datasets()
    rows = verify_e01(datasets) + verify_e02(datasets)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "e0_verificacion_empates.csv", index=False)
    if len(table):
        for name, group in table.groupby("experimento"):
            print(f"  {name}: {int(group['exact_equal'].sum())}/"
                  f"{len(group)} casos coinciden tras desplazar eps")
    else:
        print("  no había discrepancias que verificar")
