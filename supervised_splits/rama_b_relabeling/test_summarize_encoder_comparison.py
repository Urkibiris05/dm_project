"""Tests for encoder comparison aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from summarize_encoder_comparison import build_comparison


def test_build_comparison_reads_run_and_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "exploratory-test-common-j10-k5"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "model_id": "test/model",
                "disorder": "Ansiedad",
                "n_messages": 10,
                "initial_suffer": 6,
                "initial_control": 4,
                "relabelled_suffer_to_control": 3,
                "final_suffer": 3,
                "final_control": 7,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "trastorno": "Ansiedad",
                "share_positive_subjects_affected": 0.5,
            }
        ]
    ).to_csv(run_dir / "summary_by_disorder.csv", index=False)

    comparison = build_comparison(tmp_path)

    assert comparison.loc[0, "share_initial_suffer_relabelled"] == 0.5
    assert comparison.loc[0, "positive_subjects_affected"] == 0.5


def test_build_comparison_rejects_mismatched_disorder(tmp_path: Path) -> None:
    run_dir = tmp_path / "exploratory-test-common-j10-k5"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"disorder": "TCA"}),
        encoding="utf-8",
    )
    pd.DataFrame([{"trastorno": "Ansiedad"}]).to_csv(
        run_dir / "summary_by_disorder.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="El trastorno no coincide"):
        build_comparison(tmp_path)
