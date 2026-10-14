"""Tests de unidades, hora, combinación, caché y nombres."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vectorization.cache import VectorCache
from vectorization.cli import build_parser
from vectorization.combine import combine_blocks
from vectorization.naming import build_name
from vectorization.time_features import message_time_matrix
from vectorization.units import aggregate_rows, build_units, split_blocks


def test_aggregate_rows_mean_and_max() -> None:
    """Agrega por unidad filas contiguas."""
    matrix = np.array([[1.0, 4.0], [3.0, 2.0], [10.0, 10.0]])
    owner = np.array([0, 0, 1])
    assert aggregate_rows(matrix, owner, "mean").tolist() == [
        [2.0, 3.0], [10.0, 10.0]]
    assert aggregate_rows(matrix, owner, "max").tolist() == [
        [3.0, 4.0], [10.0, 10.0]]


def test_split_blocks_merges_tiny_remainder() -> None:
    """Un resto menor que la mitad del bloque se une al anterior."""
    text = " ".join(str(i) for i in range(23))
    blocks = split_blocks(text, 10)
    assert [len(block.split()) for block in blocks] == [10, 13]
    assert split_blocks("", 10) == [""]


def test_user_units_concatenate_in_date_order(corpus_root: Path) -> None:
    """El texto de usuario concatena sus mensajes y cuenta cuántos son."""
    from vectorization.corpus import load_corpus
    df = load_corpus(corpus_root, ["Depresion"], ["Train"])
    units = build_units(df, "user")
    assert len(units.texts) == 2
    assert units.metadata["n_messages"].tolist() == [5, 3]
    assert units.texts[1].startswith("me encanta salir a correr")
    assert units.owner.tolist() == [0, 0, 0, 0, 0, 1, 1, 1]


def test_hour_encoding_is_cyclic() -> None:
    """Las 23:59 quedan más cerca de las 00:01 que de las 12:00."""
    dates = pd.Series(pd.to_datetime([
        "2021-01-04 23:59:00", "2021-01-04 00:01:00",
        "2021-01-04 12:00:00"]))
    cyclic = message_time_matrix(dates)[:, :2]
    near = np.linalg.norm(cyclic[0] - cyclic[1])
    far = np.linalg.norm(cyclic[0] - cyclic[2])
    assert near < 0.1 < far


def test_combine_blocks_respects_weights() -> None:
    """Cada bloque tiene por norma su peso (el semántico, 1 - resto)."""
    rng = np.random.default_rng(0)
    semantic = rng.normal(size=(6, 5))
    affect = rng.random(size=(6, 3))
    time_block = rng.normal(size=(6, 4))
    combined = combine_blocks(semantic, affect, time_block, 0.2, 0.1)
    assert combined.shape == (6, 12)
    assert np.allclose(np.linalg.norm(combined[:, :5], axis=1), 0.7,
                       atol=1e-6)
    assert np.allclose(np.linalg.norm(combined[:, 5:8], axis=1), 0.2,
                       atol=1e-6)
    assert np.allclose(np.linalg.norm(combined[:, 8:], axis=1), 0.1,
                       atol=1e-6)


def test_combine_without_semantic_rescales_weights() -> None:
    """En 'features' los pesos se reescalan para sumar 1."""
    rng = np.random.default_rng(1)
    affect = rng.random((4, 3))
    time_block = rng.normal(size=(4, 2))
    combined = combine_blocks(None, affect, time_block, 0.3, 0.1)
    assert np.allclose(np.linalg.norm(combined[:, :3], axis=1), 0.75,
                       atol=1e-6)
    assert np.allclose(np.linalg.norm(combined[:, 3:], axis=1), 0.25,
                       atol=1e-6)


def test_cache_only_computes_missing_texts(tmp_path: Path) -> None:
    """La segunda petición reutiliza lo ya calculado, también tras recargar."""
    calls: list[list[str]] = []

    def compute(texts: list[str]) -> np.ndarray:
        calls.append(texts)
        return np.array([[float(len(text))] for text in texts])

    path = tmp_path / "cache.npz"
    cache = VectorCache(path)
    first = cache.get_matrix(["aa", "b", "aa"], compute)
    assert first.ravel().tolist() == [2.0, 1.0, 2.0]
    assert calls == [["aa", "b"]]

    reloaded = VectorCache(path)
    second = reloaded.get_matrix(["b", "cccc"], compute)
    assert second.ravel().tolist() == [1.0, 4.0]
    assert calls[-1] == ["cccc"]
    assert (reloaded.hits, reloaded.misses) == (1, 1)

    disabled = VectorCache(path, enabled=False)
    disabled.get_matrix(["b"], compute)
    assert calls[-1] == ["b"]


def parse(*argv: str) -> argparse.Namespace:
    """Interpreta argumentos con el analizador real de la línea de comandos."""
    return build_parser().parse_args(list(argv))


def test_folder_names_are_deterministic() -> None:
    """El nombre refleja lo que se aparta de los valores por defecto."""
    assert build_name(parse("tfidf", "--analyzer", "char_wb")) == (
        "message__all__tfidf-char_wb-1-5-svd128")
    args = parse("transformer", "--model", "minilm", "--unit", "user",
                 "--disorders", "Depresion", "--affect", "0.2",
                 "--time", "0.1")
    assert build_name(args) == (
        "user__Depresion__minilm-mean__affect0.2-time0.1")
    args = parse("transformer", "--model", "e5", "--fit-on", "train",
                 "--emojis", "remove")
    assert build_name(args) == "message__all__e5__fit-train-noemoji"
