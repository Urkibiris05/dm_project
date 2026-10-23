"""Tests de la carga del corpus y de la capa de preprocesado."""

import json
from pathlib import Path

from vectorization.corpus import load_corpus
from vectorization.preprocessing import (build_emoji_index, preprocess,
                                         strip_emoji_names)

EMOJI_NAMES = ["cara llorando de risa", "mano saludando",
               "mano saludando tono de piel medio"]


def test_subject_keys_are_unique_across_disorders(corpus_root: Path) -> None:
    """subject1 existe en Depresion y en TCA y no debe mezclarse."""
    df = load_corpus(corpus_root, ["Depresion", "TCA"], ["Train"])
    assert df["subject_key"].nunique() == 3
    assert "TCA/Train/subject1" in set(df["subject_key"])
    assert set(df["label"]) == {0, 1}


def test_load_corpus_accepts_accented_depression_and_subject_separator(
        tmp_path: Path) -> None:
    """La carga acepta variantes habituales del formato real del corpus."""
    split_dir = tmp_path / "Depresión" / "Train"
    split_dir.mkdir(parents=True)
    (split_dir / "subject_1.json").write_text(json.dumps({
        "messages": [
            {"id": 7, "timestamp": "2021-01-01T10:00:00",
             "text": "primer mensaje"},
        ],
    }), encoding="utf-8")

    df = load_corpus(tmp_path, ["Depresion"], ["Train"])

    assert len(df) == 1
    assert df.at[0, "disorder"] == "Depresion"
    assert df.at[0, "message"] == "primer mensaje"


def test_strip_emoji_prefers_longest_name() -> None:
    """Quita el nombre más largo y deja el resto del texto."""
    index = build_emoji_index(EMOJI_NAMES)
    text = "hola mano saludando tono de piel medio adiós"
    assert strip_emoji_names(text, index) == ("hola adiós", 1)
    assert strip_emoji_names("sin emojis aquí", index) == (
        "sin emojis aquí", 0)


def test_short_messages_are_dropped_without_counting_emojis(
        corpus_root: Path) -> None:
    """'jaja cara llorando de risa' tiene 1 token real y se descarta."""
    df = load_corpus(corpus_root, ["Depresion"], ["Train"])
    index = build_emoji_index(EMOJI_NAMES)
    kept, report = preprocess(df, 4, False, False, "keep", index)
    assert "jaja cara llorando de risa" not in set(kept["message"])
    assert "ok" not in set(kept["message"])
    assert report["dropped_short"] == 2

    kept, report = preprocess(df, 4, True, False, "keep", index)
    assert "jaja cara llorando de risa" in set(kept["message"])
    assert report["dropped_short"] == 1


def test_duplicates_and_emoji_removal(corpus_root: Path) -> None:
    """Se quita el duplicado exacto; con 'remove' no quedan nombres."""
    df = load_corpus(corpus_root, ["Depresion"], ["Train"])
    index = build_emoji_index(EMOJI_NAMES)
    kept, report = preprocess(df, 3, True, False, "remove", index)
    assert report["dropped_duplicates"] == 1
    assert report["dropped_empty"] == 0
    assert all("llorando" not in text for text in kept["message"])
    kept, report = preprocess(df, 3, True, True, "keep", index)
    assert report["dropped_duplicates"] == 0


def test_subject_without_messages_disappears(corpus_root: Path) -> None:
    """Con un mínimo muy alto nadie conserva mensajes."""
    df = load_corpus(corpus_root, ["TCA"], ["Train"])
    kept, report = preprocess(df, 50, False, False, "keep", {})
    assert len(kept) == 0
    assert report["subjects_in"] == 1
    assert report["subjects_out"] == 0
