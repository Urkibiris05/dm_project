"""Tests for the train-only relabeling runner."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from relabeling import RelabelRound, RelabelResult
import run_relabeling
from run_relabeling import load_aligned_train_data, write_run


def _train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": "positive",
                "id_message": "m1",
                "label": 1,
                "partition": "train",
                "trastorno": "Ansiedad",
            },
            {
                "subject_id": "control",
                "id_message": "m2",
                "label": 0,
                "partition": "train",
                "trastorno": "Ansiedad",
            },
        ]
    )


def test_loader_accepts_aligned_train_cache(tmp_path) -> None:
    messages = _train_frame()
    csv_path = tmp_path / "messages_train.csv"
    cache_path = tmp_path / "embeddings.npz"
    messages.to_csv(csv_path, index=False)
    np.savez(
        cache_path,
        embeddings=np.eye(2),
        subject_id=messages["subject_id"].to_numpy(dtype=str),
        id_message=messages["id_message"].to_numpy(dtype=str),
        disorder=np.array("Ansiedad"),
        model_id=np.array("test-encoder"),
    )

    loaded_messages, vectors, model_id = load_aligned_train_data(
        csv_path,
        cache_path,
        "Ansiedad",
    )

    assert loaded_messages["id_message"].tolist() == ["m1", "m2"]
    assert vectors.shape == (2, 2)
    assert model_id == "test-encoder"


def test_loader_rejects_misaligned_cache(tmp_path) -> None:
    messages = _train_frame()
    csv_path = tmp_path / "messages_train.csv"
    cache_path = tmp_path / "embeddings.npz"
    messages.to_csv(csv_path, index=False)
    np.savez(
        cache_path,
        embeddings=np.eye(2),
        subject_id=messages["subject_id"].to_numpy(dtype=str),
        id_message=np.array(["m2", "m1"]),
        disorder=np.array("Ansiedad"),
    )

    with pytest.raises(ValueError, match="id_message"):
        load_aligned_train_data(csv_path, cache_path, "Ansiedad")


def test_loader_rejects_non_train_csv(tmp_path) -> None:
    messages = _train_frame()
    messages.loc[0, "partition"] = "dev"
    csv_path = tmp_path / "messages_train.csv"
    cache_path = tmp_path / "embeddings.npz"
    messages.to_csv(csv_path, index=False)
    np.savez(
        cache_path,
        embeddings=np.eye(2),
        subject_id=messages["subject_id"].to_numpy(dtype=str),
        id_message=messages["id_message"].to_numpy(dtype=str),
        disorder=np.array("Ansiedad"),
    )

    with pytest.raises(ValueError, match="únicamente filas de train"):
        load_aligned_train_data(csv_path, cache_path, "Ansiedad")


def test_write_run_does_not_export_message_text(tmp_path) -> None:
    messages = _train_frame()
    messages["message"] = ["private text one", "private text two"]
    result = RelabelResult(
        labels=np.array([0, 0]),
        history=(
            RelabelRound(
                round_number=1,
                changed_this_round=1,
                changed_total=1,
                remaining_suffer=0,
                total_control=2,
            ),
        ),
    )
    output_dir = tmp_path / "run"

    write_run(
        messages,
        result,
        output_dir=output_dir,
        disorder="Ansiedad",
        model_id="test-encoder",
        neighbors=1,
        min_control_neighbors=1,
        mode="single_pass",
        exclude_same_subject=False,
        max_neighbors_per_subject=None,
        max_rounds=None,
    )

    labels = pd.read_csv(output_dir / "weak_message_labels.csv")
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert labels["was_relabelled"].tolist() == [True, False]
    assert "message" not in labels.columns
    assert "private text" not in (
        output_dir / "weak_message_labels.csv"
    ).read_text(encoding="utf-8")
    assert manifest["relabelled_suffer_to_control"] == 1
    assert "not human-validated" in manifest["note"]


def test_cli_runs_and_writes_only_under_branch_reports(
    tmp_path,
    monkeypatch,
) -> None:
    messages = _train_frame()
    csv_path = tmp_path / "messages_train.csv"
    cache_path = tmp_path / "embeddings.npz"
    messages.to_csv(csv_path, index=False)
    np.savez(
        cache_path,
        embeddings=np.array([[1.0, 0.0], [0.9, 0.1]]),
        subject_id=messages["subject_id"].to_numpy(dtype=str),
        id_message=messages["id_message"].to_numpy(dtype=str),
        disorder=np.array("Ansiedad"),
        model_id=np.array("test-encoder"),
    )
    reports = tmp_path / "branch" / "reports" / "relabeling"
    monkeypatch.setattr(run_relabeling, "OUTPUT_ROOT", reports)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_relabeling.py",
            "--train-csv",
            str(csv_path),
            "--embedding-cache",
            str(cache_path),
            "--disorder",
            "Ansiedad",
            "--neighbors",
            "1",
            "--min-control-neighbors",
            "1",
            "--run-name",
            "smoke-test",
        ],
    )

    assert run_relabeling.main() == 0
    assert (reports / "smoke-test" / "weak_message_labels.csv").is_file()
    assert (reports / "smoke-test" / "run_manifest.json").is_file()
