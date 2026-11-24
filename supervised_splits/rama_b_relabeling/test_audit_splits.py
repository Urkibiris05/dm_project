"""Tests for split auditing."""

from __future__ import annotations

import pandas as pd

from audit_splits import audit_splits


def _frame(partition: str, subject_id: str = "subject-a") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": subject_id,
                "trastorno": "Ansiedad",
                "label": 1,
                "label_name": "suffer",
                "id_message": "m1",
                "message": "mensaje de prueba",
                "date": "2021-01-01 10:00:00",
                "partition": partition,
            }
        ]
    )


def test_audit_summarizes_splits_and_passes_consistent_data(tmp_path) -> None:
    for partition in ("train", "dev", "test"):
        _frame(partition, f"{partition}-subject").to_csv(
            tmp_path / f"messages_{partition}.csv",
            index=False,
        )

    summary, checks = audit_splits(tmp_path)

    assert checks["status"] == "ok"
    assert len(summary) == 3
    assert summary["messages"].tolist() == [1, 1, 1]


def test_audit_detects_subject_leakage_across_splits(tmp_path) -> None:
    _frame("train").to_csv(tmp_path / "messages_train.csv", index=False)
    _frame("dev").to_csv(tmp_path / "messages_dev.csv", index=False)
    _frame("test").to_csv(tmp_path / "messages_test.csv", index=False)

    _, checks = audit_splits(tmp_path)

    assert checks["status"] == "failed"
    assert any("multiple splits" in error for error in checks["errors"])


def test_audit_detects_label_name_mismatch(tmp_path) -> None:
    for partition in ("train", "dev", "test"):
        frame = _frame(partition)
        frame.loc[0, "label_name"] = "control"
        frame.to_csv(tmp_path / f"messages_{partition}.csv", index=False)

    _, checks = audit_splits(tmp_path)

    assert checks["status"] == "failed"
    assert any("label_name disagree" in error for error in checks["errors"])


def test_audit_counts_duplicate_text_without_copying_text_to_summary(
    tmp_path,
) -> None:
    for partition in ("train", "dev", "test"):
        frame = pd.concat(
            [
                _frame(partition, f"{partition}-subject"),
                _frame(partition, f"{partition}-subject"),
            ],
            ignore_index=True,
        )
        frame.loc[1, "id_message"] = "m2"
        frame.loc[1, "date"] = "2021-01-02 10:00:00"
        frame.to_csv(tmp_path / f"messages_{partition}.csv", index=False)

    summary, checks = audit_splits(tmp_path)

    assert checks["status"] == "ok"
    assert summary["duplicate_text_within_subject"].eq(1).all()
    assert "message" not in summary.columns
