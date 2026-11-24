"""Test alignment of previous embeddings to the current train split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from export_previous_embeddings import export_embeddings


def _write_cache(
    root: Path,
    alias: str,
    keys: list[tuple[str, str, str]],
    labels: list[int],
) -> None:
    model_dir = root / f"message__all__{alias}"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"vectorizer": {"model": f"model/{alias}"}}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "disorder": disorder,
                "subject": subject,
                "id_message": message_id,
                "label": label,
            }
            for (disorder, subject, message_id), label in zip(keys, labels)
        ]
    ).to_csv(model_dir / "metadata.csv", index=False)
    np.save(
        model_dir / "X.npy",
        np.arange(len(keys) * 2, dtype=np.float32).reshape(len(keys), 2),
    )


def _write_train(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "subject_id": "subject1",
                "trastorno": "Ansiedad",
                "source_partition": "Test",
                "label": 1,
                "label_name": "suffer",
                "class_name": "suffer",
                "id_message": "m1",
                "message": "do not export",
                "date": "2021-01-01",
                "partition": "train",
            },
            {
                "subject_id": "subject2",
                "trastorno": "TCA",
                "source_partition": "Train",
                "label": 0,
                "label_name": "control",
                "class_name": "control",
                "id_message": "m2",
                "message": "also do not export",
                "date": "2021-01-02",
                "partition": "train",
            },
        ]
    ).to_csv(path, index=False)


def test_exports_only_common_train_keys_without_message_text(
    tmp_path: Path,
) -> None:
    train_csv = tmp_path / "train.csv"
    embedding_root = tmp_path / "vectors"
    output_dir = tmp_path / "export"
    _write_train(train_csv)
    train_keys = [
        ("Ansiedad", "subject1", "m1"),
        ("TCA", "subject2", "m2"),
    ]
    _write_cache(embedding_root, "model-a", train_keys, [1, 0])
    _write_cache(embedding_root, "model-b", train_keys[:1], [1])

    report = export_embeddings(
        train_csv,
        embedding_root,
        output_dir,
        ("model-a", "model-b"),
    )

    common_train = pd.read_csv(output_dir / "common_train.csv")
    assert common_train["id_message"].tolist() == ["m1"]
    assert "message" not in common_train.columns
    assert report["common_train_rows"].tolist() == [1, 1]
    with np.load(output_dir / "model-a" / "Ansiedad.npz") as cache:
        assert cache["id_message"].tolist() == ["m1"]
        assert cache["embeddings"].shape == (1, 2)


def test_rejects_mismatched_subject_label(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    embedding_root = tmp_path / "vectors"
    output_dir = tmp_path / "export"
    _write_train(train_csv)
    _write_cache(
        embedding_root,
        "model-a",
        [("Ansiedad", "subject1", "m1")],
        [0],
    )

    with pytest.raises(ValueError, match="Etiqueta de sujeto discrepante"):
        export_embeddings(
            train_csv,
            embedding_root,
            output_dir,
            ("model-a",),
        )


def test_refuses_to_overwrite_export_directory(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    embedding_root = tmp_path / "vectors"
    output_dir = tmp_path / "export"
    _write_train(train_csv)
    _write_cache(
        embedding_root,
        "model-a",
        [("Ansiedad", "subject1", "m1")],
        [1],
    )
    output_dir.mkdir()

    with pytest.raises(FileExistsError):
        export_embeddings(
            train_csv,
            embedding_root,
            output_dir,
            ("model-a",),
        )
