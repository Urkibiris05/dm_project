"""Tests for text-free relabeling summaries."""

from __future__ import annotations

import pandas as pd
import pytest

from summarize_relabeling import summarize_run


def test_summary_joins_labels_and_reports_position_bands(tmp_path) -> None:
    train_csv = tmp_path / "train.csv"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    messages = pd.DataFrame(
        [
            {
                "subject_id": "a",
                "trastorno": "Ansiedad",
                "label": 1,
                "date": "2021-01-01",
                "id_message": "1",
                "partition": "train",
                "trastorno": "Ansiedad",
                "message": "do not export",
            },
            {
                "subject_id": "a",
                "trastorno": "Ansiedad",
                "label": 1,
                "date": "2021-01-02",
                "id_message": "2",
                "partition": "train",
                "trastorno": "Ansiedad",
                "message": "secret text",
            },
            {
                "subject_id": "b",
                "trastorno": "Ansiedad",
                "label": 0,
                "date": "2021-01-01",
                "id_message": "3",
                "partition": "train",
                "trastorno": "Ansiedad",
                "message": "another text",
            },
        ]
    )
    messages.to_csv(train_csv, index=False)
    pd.DataFrame(
        [
            {
                "subject_id": "a",
                "id_message": "1",
                "initial_label": 1,
                "weak_label": 0,
                "was_relabelled": True,
            },
            {
                "subject_id": "a",
                "id_message": "2",
                "initial_label": 1,
                "weak_label": 1,
                "was_relabelled": False,
            },
            {
                "subject_id": "b",
                "id_message": "3",
                "initial_label": 0,
                "weak_label": 0,
                "was_relabelled": False,
            },
        ]
    ).to_csv(run_dir / "weak_message_labels.csv", index=False)
    (run_dir / "run_manifest.json").write_text(
        '{"disorder":"Ansiedad"}',
        encoding="utf-8",
    )

    summaries = summarize_run(train_csv, run_dir, "Ansiedad")

    disorder = summaries["by_disorder"].iloc[0]
    assert disorder["initial_suffer"] == 2
    assert disorder["relabelled"] == 1
    assert disorder["share_initial_suffer_relabelled"] == 0.5
    assert disorder["positive_subjects_affected"] == 1
    assert disorder["share_positive_subjects_affected"] == 1.0
    assert set(summaries["by_position"]["position_band"]) == {
        "first_20%",
        "last_20%",
    }
    assert "message" not in summaries["by_subject"].columns


def test_summary_rejects_misaligned_labels(tmp_path) -> None:
    train_csv = tmp_path / "train.csv"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "subject_id": "a",
                "trastorno": "Ansiedad",
                "label": 1,
                "date": "2021-01-01",
                "id_message": "1",
                "partition": "train",
                "trastorno": "Ansiedad",
            }
        ]
    ).to_csv(train_csv, index=False)
    pd.DataFrame(
        [
            {
                "subject_id": "a",
                "id_message": "wrong",
                "initial_label": 1,
                "weak_label": 1,
                "was_relabelled": False,
            }
        ]
    ).to_csv(run_dir / "weak_message_labels.csv", index=False)
    (run_dir / "run_manifest.json").write_text(
        '{"disorder":"Ansiedad"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no coinciden exactamente"):
        summarize_run(train_csv, run_dir, "Ansiedad")
