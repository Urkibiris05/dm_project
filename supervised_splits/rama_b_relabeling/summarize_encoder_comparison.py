"""Aggregate comparable encoder k-NN relabeling runs into one CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = MODULE_DIR / "reports" / "relabeling"
DEFAULT_OUTPUT = (
    MODULE_DIR / "reports" / "encoder_comparison_j10_k5.csv"
)
RUN_GLOB = "exploratory-*-common-j10-k5"


def build_comparison(runs_dir: Path) -> pd.DataFrame:
    """Read run manifests and text-free per-disorder summaries."""
    rows: list[dict[str, object]] = []
    for run_dir in sorted(runs_dir.glob(RUN_GLOB)):
        manifest_path = run_dir / "run_manifest.json"
        summary_path = run_dir / "summary_by_disorder.csv"
        if not manifest_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(
                f"Falta el manifiesto o resumen en {run_dir}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = pd.read_csv(summary_path)
        if len(summary) != 1:
            raise ValueError(f"Se esperaba una fila en {summary_path}")
        disorder_summary = summary.iloc[0]
        if disorder_summary["trastorno"] != manifest["disorder"]:
            raise ValueError(
                f"El trastorno no coincide entre {manifest_path} y "
                f"{summary_path}"
            )
        rows.append(
            {
                "run": run_dir.name,
                "model_id": manifest["model_id"],
                "disorder": manifest["disorder"],
                "messages": manifest["n_messages"],
                "initial_suffer": manifest["initial_suffer"],
                "initial_control": manifest["initial_control"],
                "relabelled_suffer_to_control": (
                    manifest["relabelled_suffer_to_control"]
                ),
                "share_initial_suffer_relabelled": (
                    manifest["relabelled_suffer_to_control"]
                    / manifest["initial_suffer"]
                    if manifest["initial_suffer"]
                    else 0.0
                ),
                "final_suffer": manifest["final_suffer"],
                "final_control": manifest["final_control"],
                "positive_subjects_affected": (
                    disorder_summary["share_positive_subjects_affected"]
                ),
            }
        )
    if not rows:
        raise ValueError(f"No se encontraron runs {RUN_GLOB} en {runs_dir}")
    return pd.DataFrame(rows).sort_values(
        ["disorder", "model_id"],
        kind="stable",
    ).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    comparison = build_comparison(args.runs_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False)
    print(comparison.to_string(index=False))
    print(f"\nComparación guardada en: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
