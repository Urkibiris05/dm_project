"""Tests del flujo ``transformer`` con un codificador simulado."""

from pathlib import Path

import numpy as np
import pytest

from vectorization import transformer
from vectorization.cli import build_parser
from vectorization.corpus import load_corpus
from vectorization.units import build_units


def fake_encoder(hf_name: str, batch_size: int, truncate_dim: int | None,
                 device: str):
    """Sustituye al modelo real: el vector depende solo de la longitud."""
    def encode(texts: list[str]) -> np.ndarray:
        """Devuelve [palabras, caracteres, 1] por texto."""
        return np.array([[len(t.split()), len(t), 1.0] for t in texts])
    return encode


def forbidden_encoder(hf_name: str, batch_size: int,
                      truncate_dim: int | None, device: str):
    """Codificador que falla si se usa: comprueba que todo viene de caché."""
    def encode(texts: list[str]) -> np.ndarray:
        """No debería llamarse nunca."""
        raise AssertionError("se ha codificado algo que estaba en caché")
    return encode


@pytest.mark.parametrize("pooling", ["concat", "mean", "max", "blocks"])
def test_user_poolings_give_one_unit_vector_per_user(
        pooling: str, corpus_root: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Cada pooling devuelve una fila normalizada por usuario."""
    monkeypatch.setattr(transformer, "make_encoder", fake_encoder)
    args = build_parser().parse_args([
        "transformer", "--model", "minilm", "--unit", "user",
        "--user-pooling", pooling, "--block-words", "4"])
    df = load_corpus(corpus_root, ["Depresion"], ["Train"])
    units = build_units(df, "user")
    mask = np.ones(len(units.texts), dtype=bool)
    matrix, pca, info = transformer.vectorize(df, units, mask, args,
                                              tmp_path)
    assert matrix.shape == (2, 3)
    assert pca is None
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    assert info["cache_misses"] > 0

    # Segunda vez: todo sale de la caché, sin llamar al codificador.
    monkeypatch.setattr(transformer, "make_encoder", forbidden_encoder)
    again, _, info = transformer.vectorize(df, units, mask, args, tmp_path)
    assert info["cache_misses"] == 0
    assert np.allclose(again, matrix)


def test_e5_uses_query_prefix_and_pca_reduces_dimension(
        corpus_root: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """E5 añade 'query: ' y --pca reduce las columnas."""
    assert transformer.resolve_model("e5")[1] == "query: "
    assert transformer.resolve_model("minilm")[1] == ""
    monkeypatch.setattr(transformer, "make_encoder", fake_encoder)
    args = build_parser().parse_args([
        "transformer", "--model", "e5", "--pca", "2"])
    df = load_corpus(corpus_root, ["Depresion"], ["Train"])
    units = build_units(df, "message")
    mask = np.ones(len(units.texts), dtype=bool)
    matrix, pca, _ = transformer.vectorize(df, units, mask, args, tmp_path)
    assert matrix.shape == (len(units.texts), 2)
    assert pca is not None


@pytest.mark.parametrize(
    ("alias", "hf_name"),
    [
        ("robertuito", "pysentimiento/robertuito-base-uncased"),
        ("robertuito-base-uncased", "pysentimiento/robertuito-base-uncased"),
        ("beto-raw", "dccuchile/bert-base-spanish-wwm-uncased"),
        ("roberta-base-bne", "PlanTL-GOB-ES/roberta-base-bne"),
        ("robertalex", "PlanTL-GOB-ES/RoBERTalex"),
    ],
)
def test_spanish_transformer_aliases_resolve(
        alias: str, hf_name: str) -> None:
    """Los alias de los encoders base apuntan a sus modelos de Hugging Face."""
    assert transformer.resolve_model(alias) == (hf_name, "")


def test_roberta_bne_uses_explicit_model_loader() -> None:
    """RoBERTa-BNE conserva su alias aunque requiera carga explícita."""
    assert transformer.MODEL_ALIASES["roberta-base-bne"] == (
        "PlanTL-GOB-ES/roberta-base-bne")
