"""Audit the shared message CSVs without copying message text to the report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PARTITIONS = ("train", "dev", "test")
REQUIRED_COLUMNS = {
    "subject_id",
    "trastorno",
    "label",
    "label_name",
    "id_message",
    "message",
    "date",
    "partition",
}


def _quantile(values: pd.Series, probability: float) -> float:
    if values.empty:
        return 0.0
    return float(values.quantile(probability))


def audit_splits(input_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize class, subject, message, and chronology integrity."""
    frames: dict[str, pd.DataFrame] = {}
    checks: dict[str, Any] = {"errors": [], "warnings": []}

    for partition in PARTITIONS:
        path = input_dir / f"messages_{partition}.csv"
        if not path.is_file():
            checks["errors"].append(f"Missing file: {path}")
            continue
        frame = pd.read_csv(
            path,
            dtype={"subject_id": "string", "id_message": "string"},
        )
        missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing_columns:
            checks["errors"].append(
                f"{path.name} missing columns: {missing_columns}"
            )
            continue
        frames[partition] = frame

    if checks["errors"]:
        return pd.DataFrame(), checks

    rows: list[dict[str, Any]] = []
    subject_partitions: dict[tuple[str, str], set[str]] = {}
    for partition, frame in frames.items():
        frame["date_parsed"] = pd.to_datetime(frame["date"], errors="coerce")
        for (disorder, label), group in frame.groupby(
            ["trastorno", "label"], dropna=False, sort=True
        ):
            by_subject = group.groupby("subject_id", dropna=False)
            message_counts = by_subject.size()
            nonempty = group["message"].fillna("").astype("string").str.strip()
            rows.append(
                {
                    "partition": partition,
                    "trastorno": disorder,
                    "label": label,
                    "label_name": group["label_name"].iloc[0],
                    "subjects": int(group["subject_id"].nunique(dropna=False)),
                    "messages": int(len(group)),
                    "messages_per_subject_min": int(message_counts.min()),
                    "messages_per_subject_median": _quantile(
                        message_counts, 0.5
                    ),
                    "messages_per_subject_p95": _quantile(
                        message_counts, 0.95
                    ),
                    "messages_per_subject_max": int(message_counts.max()),
                    "empty_messages": int(nonempty.eq("").sum()),
                    "missing_dates": int(group["date_parsed"].isna().sum()),
                    "duplicate_message_ids": int(
                        group.duplicated(["subject_id", "id_message"]).sum()
                    ),
                    "duplicate_text_within_subject": int(
                        group.assign(
                            normalized_message=nonempty.str.casefold()
                        ).duplicated(
                            ["subject_id", "normalized_message"]
                        ).sum()
                    ),
                    "duplicate_subject_timestamps": int(
                        group.duplicated(
                            ["subject_id", "date_parsed"], keep=False
                        ).sum()
                    ),
                    "message_chars_median": _quantile(
                        nonempty.str.len().dropna(), 0.5
                    ),
                    "message_chars_p95": _quantile(
                        nonempty.str.len().dropna(), 0.95
                    ),
                }
            )
        for subject_id, subject_group in frame.groupby(
            "subject_id", dropna=False
        ):
            key = (str(subject_group["trastorno"].iloc[0]), str(subject_id))
            subject_partitions.setdefault(key, set()).add(partition)

        invalid_labels = frame["label"].isna().any() or not frame[
            "label"
        ].isin([0, 1]).all()
        if invalid_labels:
            checks["errors"].append(
                f"{partition}: label contains null or non-binary values"
            )
        expected_names = frame["label"].map({0: "control", 1: "suffer"})
        if not expected_names.eq(frame["label_name"]).all():
            checks["errors"].append(
                f"{partition}: label and label_name disagree"
            )
        if frame["partition"].astype("string").ne(partition).any():
            checks["errors"].append(
                f"{partition}: row partition column does not match file"
            )
        missing_ids = (
            frame["subject_id"].isna().any()
            or frame["id_message"].isna().any()
        )
        if missing_ids:
            checks["errors"].append(
                f"{partition}: null subject_id or id_message"
            )

    crossing_subjects = [
        {
            "trastorno": disorder,
            "subject_id": subject_id,
            "partitions": sorted(parts),
        }
        for (disorder, subject_id), parts in subject_partitions.items()
        if len(parts) > 1
    ]
    if crossing_subjects:
        checks["errors"].append(
            f"{len(crossing_subjects)} subject(s) appear in multiple splits"
        )
        checks["subjects_in_multiple_splits"] = crossing_subjects[:20]

    summary = pd.DataFrame(rows).sort_values(
        ["trastorno", "label", "partition"]
    )
    checks["files"] = {
        partition: {
            "rows": int(len(frame)),
            "subjects": int(
                frame[["trastorno", "subject_id"]].drop_duplicates().shape[0]
            ),
        }
        for partition, frame in frames.items()
    }
    checks["status"] = "ok" if not checks["errors"] else "failed"
    return summary, checks


def main() -> int:
    default_input = Path(__file__).resolve().parent.parent / "resultados"
    default_output = Path(__file__).resolve().parent / "reports"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    args = parser.parse_args()

    summary, checks = audit_splits(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "split_audit.csv"
    checks_path = args.output_dir / "split_audit_checks.json"
    summary.to_csv(summary_path, index=False)
    checks_path.write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Audit status: {checks['status']}")
    print(f"Summary: {summary_path}")
    print(f"Checks: {checks_path}")
    if checks["errors"]:
        for error in checks["errors"]:
            print(f"ERROR: {error}")
        return 1
    if not summary.empty:
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
