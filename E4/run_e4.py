"""Ejecución del experimento E4 sobre una vectorización concreta."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import MIN_SAMPLES_VALUES, P_VALUES, RESULTS_DIR
from .exp_minkowski import build_experiment_table, resolve_vectorization_dir


def _resolve_vectorization_names(requested: list[str]) -> list[str]:
    """Resuelve nombres de vectorización, incluyendo el caso especial 'all'."""
    names: list[str] = []
    for item in requested:
        if item == "all":
            vectorizations_dir = Path("vectorizations")
            if vectorizations_dir.exists():
                names.extend(
                    sorted(
                        child.name
                        for child in vectorizations_dir.iterdir()
                        if child.is_dir() and child.name.startswith("message__")
                    )
                )
            continue
        names.append(item)
    if not names:
        raise ValueError("No se indicó ninguna vectorización para evaluar.")
    return names


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


def _filename_slug(name: str) -> str:
    """Normaliza un nombre para usarlo en el nombre de archivo."""
    return (
        str(name)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


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
        nargs="+",
        default=["message__all__minilm"],
        help=(
            "Nombre(s) o ruta(s) de la(s) vectorización(es) a evaluar. "
            "Ejemplo: --vectorization message__all__minilm "
            "message__all__beto-raw. También puedes usar --vectorization all."
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        nargs="+",
        default=list(MIN_SAMPLES_VALUES),
        help=(
            "Valores de min_samples para evaluar. Se puede pasar uno o varios, "
            "por ejemplo: --min-samples 3 5 10 20."
        ),
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

    vectorization_names = _resolve_vectorization_names(args.vectorization)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(vectorization_names) == 1:
        vectorization_name = vectorization_names[0]
        vectorization_dir = resolve_vectorization_dir(vectorization_name)
        detail_table = pd.DataFrame()
        for min_samples in args.min_samples:
            detail_table = pd.concat(
                [
                    detail_table,
                    build_experiment_table(
                        vectorization_dir,
                        min_samples=min_samples,
                        max_samples=args.max_samples,
                        p_values=tuple(args.p_values),
                        normalized_flags=(False, True),
                        seed=args.seed,
                    ),
                ],
                ignore_index=True,
            )

        detail_path = output_dir / "e4_minkowski_detail.csv"
        summary_path = output_dir / "e4_minkowski_summary.csv"
        detail_table.to_csv(detail_path, index=False)
        _build_summary(detail_table).to_csv(summary_path, index=False)
        print(f"CSV detallado: {detail_path}")
        print(f"CSV resumen: {summary_path}")
        if not detail_table.empty:
            print(_build_summary(detail_table).to_string(index=False))
        return 0

    for vectorization_name in vectorization_names:
        vectorization_dir = resolve_vectorization_dir(vectorization_name)
        detail_table = pd.DataFrame()
        for min_samples in args.min_samples:
            detail_table = pd.concat(
                [
                    detail_table,
                    build_experiment_table(
                        vectorization_dir,
                        min_samples=min_samples,
                        max_samples=args.max_samples,
                        p_values=tuple(args.p_values),
                        normalized_flags=(False, True),
                        seed=args.seed,
                    ),
                ],
                ignore_index=True,
            )

        slug = _filename_slug(vectorization_name)
        detail_path = output_dir / f"e4_minkowski_detail_{slug}.csv"
        summary_path = output_dir / f"e4_minkowski_summary_{slug}.csv"
        detail_table.to_csv(detail_path, index=False)
        _build_summary(detail_table).to_csv(summary_path, index=False)
        print(f"CSV detallado: {detail_path}")
        print(f"CSV resumen: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
