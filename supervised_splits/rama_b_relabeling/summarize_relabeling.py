"""Summarize weak-label changes by disorder, subject, and message position."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_CSV = MODULE_DIR.parent / "resultados" / "messages_train.csv"
DISORDERS = ("Ansiedad", "Depresion", "TCA")


def summarize_run(
    train_csv: Path,
    run_dir: Path,
    disorder: str,
) -> dict[str, pd.DataFrame]:
    """Join weak labels to train metadata and make text-free summaries."""
    messages = pd.read_csv(
        train_csv,
        dtype={"subject_id": "string", "id_message": "string"},
    )
    labels_path = run_dir / "weak_message_labels.csv"
    weak_labels = pd.read_csv(
        labels_path,
        dtype={"subject_id": "string", "id_message": "string"},
    )
    required_messages = {
        "subject_id",
        "trastorno",
        "label",
        "date",
        "partition",
        "trastorno",
    }
    required_labels = {
        "subject_id",
        "id_message",
        "initial_label",
        "weak_label",
        "was_relabelled",
    }
    missing_messages = sorted(required_messages - set(messages.columns))
    missing_labels = sorted(required_labels - set(weak_labels.columns))
    if missing_messages:
        raise ValueError(f"CSV train sin columnas: {missing_messages}")
    if missing_labels:
        raise ValueError(f"Etiquetas débiles sin columnas: {missing_labels}")
    if not messages["partition"].astype("string").eq("train").all():
        raise ValueError("El CSV de entrada debe contener solo train")
    messages = messages.loc[messages["trastorno"].eq(disorder)].reset_index(
        drop=True
    )
    if messages.empty:
        raise ValueError(f"No hay mensajes train para el trastorno {disorder}")
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("disorder") != disorder:
        raise ValueError(
            "El trastorno del run_manifest no coincide con el solicitado"
        )
    if messages.duplicated(["subject_id", "id_message"]).any():
        raise ValueError("IDs de mensaje duplicados en el CSV de train")
    if weak_labels.duplicated(["subject_id", "id_message"]).any():
        raise ValueError("IDs de mensaje duplicados en las etiquetas débiles")

    merged = messages[
        [
            "subject_id",
            "trastorno",
            "label",
            "date",
            "id_message",
        ]
    ].merge(
        weak_labels[
            [
                "subject_id",
                "id_message",
                "initial_label",
                "weak_label",
                "was_relabelled",
            ]
        ],
        on=["subject_id", "id_message"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(
            "Las etiquetas débiles no coinciden exactamente con los mensajes "
            "del CSV de train"
        )
    merged = merged.drop(columns="_merge")
    if not merged["label"].eq(merged["initial_label"]).all():
        raise ValueError(
            "La etiqueta inicial no coincide con la etiqueta del CSV"
        )
    if not merged["was_relabelled"].eq(
        merged["initial_label"] != merged["weak_label"]
    ).all():
        raise ValueError("was_relabelled no coincide con las etiquetas")

    merged["date_parsed"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.sort_values(
        ["subject_id", "date_parsed", "id_message"],
        kind="stable",
        na_position="last",
    )
    grouped = merged.groupby("subject_id", sort=False, dropna=False)
    merged["message_index"] = grouped.cumcount() + 1
    merged["subject_message_count"] = grouped["subject_id"].transform("size")
    denominator = (merged["subject_message_count"] - 1).replace(0, 1)
    merged["relative_position"] = (
        (merged["message_index"] - 1) / denominator
    )
    merged["position_band"] = pd.cut(
        merged["relative_position"],
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001],
        labels=["first_20%", "20-40%", "40-60%", "60-80%", "last_20%"],
    ).astype("string")

    positive = merged["initial_label"].eq(1)
    merged["initial_positive"] = positive
    merged["relabelled_positive"] = positive & merged["was_relabelled"]

    by_disorder = (
        merged.groupby("trastorno", sort=True, observed=True)
        .agg(
            messages=("id_message", "size"),
            initial_suffer=("initial_positive", "sum"),
            relabelled=("relabelled_positive", "sum"),
            initial_control=(
                "initial_label",
                lambda values: (values == 0).sum(),
            ),
            final_suffer=("weak_label", lambda values: (values == 1).sum()),
            final_control=("weak_label", lambda values: (values == 0).sum()),
            subjects=("subject_id", "nunique"),
        )
        .reset_index()
    )
    by_disorder["share_initial_suffer_relabelled"] = (
        by_disorder["relabelled"] / by_disorder["initial_suffer"]
    )

    by_position = (
        merged.loc[positive]
        .groupby(["trastorno", "position_band"], sort=True, observed=True)
        .agg(
            initial_suffer=("id_message", "size"),
            relabelled=("relabelled_positive", "sum"),
        )
        .reset_index()
    )
    by_position["share_relabelled"] = (
        by_position["relabelled"] / by_position["initial_suffer"]
    )

    by_subject = (
        merged.groupby(
            ["trastorno", "subject_id", "label"],
            sort=True,
            observed=True,
        )
        .agg(
            messages=("id_message", "size"),
            initial_suffer=("initial_positive", "sum"),
            relabelled=("relabelled_positive", "sum"),
        )
        .reset_index()
    )
    by_subject["share_suffer_relabelled"] = (
        by_subject["relabelled"]
        / by_subject["initial_suffer"].replace(0, pd.NA)
    ).fillna(0.0)
    positive_subjects = by_subject.loc[by_subject["label"].eq(1)].copy()
    positive_subjects["was_affected"] = positive_subjects["relabelled"] > 0
    subject_effect = (
        positive_subjects.groupby("trastorno", sort=True, observed=True)
        .agg(
            positive_subjects=("subject_id", "size"),
            positive_subjects_affected=("was_affected", "sum"),
            median_relabelled_share_per_positive_subject=(
                "share_suffer_relabelled",
                "median",
            ),
        )
        .reset_index()
    )
    subject_effect["share_positive_subjects_affected"] = (
        subject_effect["positive_subjects_affected"]
        / subject_effect["positive_subjects"]
    )
    by_disorder = by_disorder.merge(
        subject_effect,
        on="trastorno",
        how="left",
        validate="one_to_one",
    )
    return {
        "by_disorder": by_disorder,
        "by_position": by_position,
        "by_subject": by_subject,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--disorder",
        choices=DISORDERS,
        required=True,
    )
    args = parser.parse_args()
    summaries = summarize_run(args.train_csv, args.run_dir, args.disorder)
    output_paths: list[Path] = []
    for name, summary in summaries.items():
        output_path = args.run_dir / f"summary_{name}.csv"
        summary.to_csv(output_path, index=False)
        output_paths.append(output_path)
    for name in ("by_disorder", "by_position"):
        print(f"\n{name}:")
        print(summaries[name].to_string(index=False))
    print("\nWrote:")
    for output_path in output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
