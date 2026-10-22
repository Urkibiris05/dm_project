"""Ejecución del experimento E4 sobre una vectorización concreta."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import P_VALUES, RESULTS_DIR
from .exp_minkowski import build_experiment_table, resolve_vectorization_dir


def _build_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Selecciona la mejor fila por configuración p + normalización."""
    if table.empty:
        return table

    rows: list[dict[str, object]] = []
    for _, group in table.groupby(["p", "normalized"], sort=True):
        best_row = group.sort_values(
            ["dbcv", "noise_rate"],
            ascending=[False, True],
            na_position="last",
        ).iloc[0]
        rows.append(best_row.to_dict())
    return pd.DataFrame(rows).sort_values(
        ["p", "normalized"],
        ascending=[True, True],
    ).reset_index(drop=True)


def main() -> int:
    """Ejecuta E4 y guarda dos CSVs: detallado y resumido."""
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta el experimento E4 para comparar distancias Minkowski "
            "con distintos exponentes p y con/ sin normalización L2."
        )
    )
    parser.add_argument(
        "--vectorization",
        default="message__all__minilm",
        help=(
            "Nombre o ruta de la vectorización a evaluar. Ejemplo: "
            "message__all__minilm"
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Número mínimo de puntos de vecindario para DBSCAN.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Máximo de instancias a tomar para controlar el coste O(n²).",
    )
    parser.add_argument(
        "--p-values",
        nargs="+",
        type=float,
        default=list(P_VALUES),
        help="Exponentes de Minkowski a evaluar.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Semilla para el submuestreo aleatorio.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directorio donde guardar los CSVs de salida.",
    )
    args = parser.parse_args()

    vectorization_dir = resolve_vectorization_dir(args.vectorization)
    detail_table = build_experiment_table(
        vectorization_dir,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        p_values=tuple(args.p_values),
        normalized_flags=(False, True),
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "e4_minkowski_detail.csv"
    summary_path = output_dir / "e4_minkowski_summary.csv"
    detail_table.to_csv(detail_path, index=False)
    _build_summary(detail_table).to_csv(summary_path, index=False)

    print(f"CSV detallado: {detail_path}")
    print(f"CSV resumen: {summary_path}")
    if not detail_table.empty:
        print(_build_summary(detail_table).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
