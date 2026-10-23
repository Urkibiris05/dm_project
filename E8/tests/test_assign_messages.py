"""Tests for persisting a DBSCAN model and assigning TXT messages."""

import json

import joblib
import numpy as np
import pandas as pd

from E8 import assign_messages


def make_representation(tmp_path):
    folder = tmp_path / "vectors" / "message__all__beto-sts"
    folder.mkdir(parents=True)
    angles = np.array([0.0, 0.03, 2.0, 2.03, 1.0])
    matrix = np.column_stack((np.cos(angles), np.sin(angles)))
    np.save(folder / "X.npy", matrix)
    pd.DataFrame({
        "split": ["Train", "Train", "Train", "Train", "Test"],
    }).to_csv(folder / "metadata.csv", index=False)
    (folder / "config.json").write_text(
        json.dumps({
            "command": "transformer",
            "args": {
                "model": "beto-sts",
                "unit": "message",
                "no_normalize": False,
                "affect": 0,
                "time": 0,
            },
        }),
        encoding="utf-8",
    )
    return folder.parent, matrix


def test_prepare_persists_train_only_model(tmp_path) -> None:
    vector_dir, _ = make_representation(tmp_path)
    model_path = tmp_path / "saved" / "model.joblib"

    artifact = assign_messages.prepare_assigner(
        "message__all__beto-sts",
        vector_dir,
        model_path,
        min_samples=2,
        percentile=100,
    )

    saved = joblib.load(model_path)
    assert artifact["n_train"] == 4
    assert artifact["n_clusters"] == 2
    assert saved["model"].n_features_in_ == 2


def test_assign_messages_uses_saved_model_and_writes_csv(
    tmp_path, monkeypatch
) -> None:
    vector_dir, matrix = make_representation(tmp_path)
    model_path = tmp_path / "model.joblib"
    assign_messages.prepare_assigner(
        "message__all__beto-sts",
        vector_dir,
        model_path,
        min_samples=2,
        percentile=100,
    )
    input_path = tmp_path / "demo.txt"
    input_path.write_text(
        "mensaje a\n\nmensaje b\nmensaje ruido\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "assigned.csv"
    vectors = matrix[[0, 2, 4]]
    monkeypatch.setattr(
        assign_messages,
        "vectorize_messages",
        lambda messages, cache_dir, batch_size, device: (
            [(1, "mensaje a"), (3, "mensaje b"), (4, "mensaje ruido")],
            vectors,
        ),
    )

    result = assign_messages.assign_messages(
        input_path,
        model_path,
        output_path,
        tmp_path / "cache",
        batch_size=4,
        device="cpu",
    )

    assert result["cluster_assigned"].tolist() == [0, 1, -1]
    assert result["input_line"].tolist() == [1, 3, 4]
    assert result["nearest_core_cluster"].tolist() == [0, 1, 0]
    assert result["distance_over_eps"].iloc[0] < 1
    assert result["distance_over_eps"].iloc[2] > 1
    assert result["assigned_cluster_train_size"].tolist() == [2, 2, 0]
    assert pd.read_csv(output_path)["is_noise"].tolist() == [
        False, False, True
    ]


def test_assign_command_prints_model_and_assignment_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    vector_dir, _ = make_representation(tmp_path)
    model_path = tmp_path / "model.joblib"
    assign_messages.prepare_assigner(
        "message__all__beto-sts",
        vector_dir,
        model_path,
        min_samples=2,
        percentile=100,
    )
    input_path = tmp_path / "input.txt"
    input_path.write_text("mensaje uno\nmensaje dos\n", encoding="utf-8")
    monkeypatch.setattr(
        assign_messages,
        "assign_messages",
        lambda *args: pd.DataFrame({
            "input_line": [1, 2],
            "text": ["mensaje uno", "mensaje dos"],
            "cluster_assigned": [0, -1],
            "is_noise": [False, True],
            "distance_over_eps": [0.5, 1.4],
            "nearest_core_cluster": [0, 1],
            "nearest_core_distance": [0.01, 0.03],
            "assigned_cluster_train_size": [2, 0],
        }),
    )

    result = assign_messages.main([
        "assign",
        "--input",
        str(input_path),
        "--model",
        str(model_path),
    ])

    output = capsys.readouterr().out
    assert result == 0
    assert "2 clusters" in output
    assert "asignados=50.0%, ruido=50.0%" in output
    assert "Distribución por cluster" in output
    assert "Asignaciones por mensaje" in output
    assert "Interpretación:" not in output


def test_missing_encoder_dependency_prints_install_command(
    tmp_path, monkeypatch, capsys
) -> None:
    input_path = tmp_path / "demo.txt"
    input_path.write_text("mensaje de prueba", encoding="utf-8")
    model_path = tmp_path / "model.joblib"
    monkeypatch.setattr(
        assign_messages,
        "assign_messages",
        lambda *args: (_ for _ in ()).throw(
            ImportError("Falta sentence-transformers")
        ),
    )

    result = assign_messages.main([
        "assign",
        "--input",
        str(input_path),
        "--model",
        str(model_path),
    ])

    captured = capsys.readouterr()
    assert result == 1
    assert "Falta sentence-transformers" in captured.err
    assert "-m pip install -r vectorization/requirements.txt" in captured.err
