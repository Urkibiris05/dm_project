"""Tests for complete BETO-STS embedding and online MLP helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baseline_mlp import (
    SplitData,
    _write_relabeling,
    choose_threshold,
    evaluate_online,
    subject_max_prefix_scores,
)
from make_beto_sts_embeddings import encode_partitions
from relabeling import RelabelResult


class FakeEncoder:
    model_id = "test-encoder"

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        assert batch_size == 2
        assert show_progress_bar
        assert convert_to_numpy
        assert not normalize_embeddings
        return np.array(
            [[len(text), 1.0] for text in texts],
            dtype=np.float32,
        )


def _split_data() -> SplitData:
    return SplitData(
        embeddings=np.ones((5, 2), dtype=np.float32),
        subject_id=np.array(["p", "p", "n", "n", "p"]),
        id_message=np.array(["2", "1", "3", "4", "3"]),
        disorder=np.array(["Ansiedad"] * 5),
        label=np.array([1, 1, 0, 0, 1]),
        date=np.array(
            [
                "2021-01-02",
                "2021-01-01",
                "2021-01-01",
                "2021-01-02",
                "2021-01-03",
            ]
        ),
    )


def test_embeddings_include_short_texts_and_do_not_write_them(
    tmp_path,
) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "embeddings"
    data_dir.mkdir()
    pd.DataFrame(
        [
            {
                "subject_id": "s1",
                "trastorno": "Ansiedad",
                "label": 1,
                "id_message": "m1",
                "message": "hi",
                "date": "2021-01-01",
                "partition": "train",
            },
            {
                "subject_id": "s2",
                "trastorno": "Ansiedad",
                "label": 0,
                "id_message": "m2",
                "message": "a longer message",
                "date": "2021-01-02",
                "partition": "train",
            },
        ]
    ).to_csv(data_dir / "messages_train.csv", index=False)

    manifest = encode_partitions(
        data_dir=data_dir,
        output_dir=output_dir,
        partitions=("train",),
        batch_size=2,
        encoder=FakeEncoder(),
    )

    with np.load(output_dir / "train.npz", allow_pickle=False) as archive:
        assert archive["embeddings"].shape == (2, 2)
        assert archive["id_message"].tolist() == ["m1", "m2"]
        assert "message" not in archive.files
    assert manifest["splits"]["train"]["messages"] == 2


def test_cumulative_mean_uses_chronological_message_order() -> None:
    data = _split_data()
    probabilities = np.array([0.9, 0.1, 0.2, 0.4, 0.8])

    scores = subject_max_prefix_scores(probabilities, data, "Ansiedad")

    positive = scores.loc[scores["subject_id"].eq("p")].iloc[0]
    np.testing.assert_allclose(positive["prefix_means"], [0.1, 0.5, 0.6])
    assert positive["max_prefix_mean"] == 0.6


def test_threshold_and_online_metrics_are_subject_level() -> None:
    data = _split_data()
    probabilities = np.array([0.9, 0.1, 0.2, 0.4, 0.8])
    scores = subject_max_prefix_scores(probabilities, data, "Ansiedad")

    threshold, selected = choose_threshold(scores)
    outcomes, metrics = evaluate_online(
        probabilities,
        data,
        "Ansiedad",
        threshold,
    )

    assert 0.0 <= threshold <= 1.0
    assert selected["macro_f1"] == 1.0
    assert len(outcomes) == 2
    assert metrics["macro_f1"] == 1.0
    assert metrics["positive_subject_detection_coverage"] == 1.0


def test_relabeling_policies_keep_separate_round_histories(tmp_path) -> None:
    data = _split_data()
    result = RelabelResult(
        labels=data.label.copy(),
        history=(),
    )
    include_path = tmp_path / "weak_labels_include_same_subject.csv"
    exclude_path = tmp_path / "weak_labels_exclude_same_subject.csv"

    _write_relabeling(data, "Ansiedad", result, include_path)
    _write_relabeling(data, "Ansiedad", result, exclude_path)

    assert (
        tmp_path
        / "round_history_weak_labels_include_same_subject.csv"
    ).is_file()
    assert (
        tmp_path
        / "round_history_weak_labels_exclude_same_subject.csv"
    ).is_file()
