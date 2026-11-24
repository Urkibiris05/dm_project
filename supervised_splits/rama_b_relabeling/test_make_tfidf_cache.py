"""Tests for the train-only TF-IDF/LSA cache builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from make_tfidf_cache import build_tfidf_cache


def _messages() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": "positive",
                "id_message": "m1",
                "message": "ansiedad intensa y miedo constante",
                "partition": "train",
                "trastorno": "Ansiedad",
            },
            {
                "subject_id": "control",
                "id_message": "m2",
                "message": "hoy hace buen tiempo para caminar",
                "partition": "train",
                "trastorno": "Ansiedad",
            },
            {
                "subject_id": "positive",
                "id_message": "m3",
                "message": "me siento con miedo y mucha ansiedad",
                "partition": "train",
                "trastorno": "Ansiedad",
            },
        ]
    )


def test_cache_builder_preserves_row_alignment_and_model_metadata(
    tmp_path,
) -> None:
    csv_path = tmp_path / "train.csv"
    output_path = tmp_path / "embeddings.npz"
    messages = _messages()
    messages.to_csv(csv_path, index=False)

    metadata = build_tfidf_cache(
        csv_path,
        output_path,
        disorder="Ansiedad",
        components=2,
        random_state=17,
        max_features=100,
        min_df=1,
    )

    with np.load(output_path, allow_pickle=False) as cache:
        assert cache["embeddings"].shape == (3, 2)
        assert cache["subject_id"].astype(str).tolist() == [
            "positive",
            "control",
            "positive",
        ]
        assert cache["id_message"].astype(str).tolist() == ["m1", "m2", "m3"]
        assert cache["disorder"].item() == "Ansiedad"
        assert "tfidf" in cache["model_id"].item()
    assert metadata["fitted_on"] == "train only"
    assert "not a Sentence Transformer" in metadata["interpretation"]
    assert output_path.with_suffix(".json").is_file()


def test_cache_builder_rejects_non_train_rows(tmp_path) -> None:
    messages = _messages()
    messages.loc[0, "partition"] = "dev"
    csv_path = tmp_path / "mixed.csv"
    messages.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Solo se permite"):
        build_tfidf_cache(
            csv_path,
            tmp_path / "embeddings.npz",
            disorder="Ansiedad",
            components=2,
            random_state=17,
            max_features=100,
            min_df=1,
        )


def test_cache_builder_refuses_to_overwrite(tmp_path) -> None:
    csv_path = tmp_path / "train.csv"
    output_path = tmp_path / "embeddings.npz"
    _messages().to_csv(csv_path, index=False)
    output_path.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="no se sobrescribirá"):
        build_tfidf_cache(
            csv_path,
            output_path,
            disorder="Ansiedad",
            components=2,
            random_state=17,
            max_features=100,
            min_df=1,
        )
