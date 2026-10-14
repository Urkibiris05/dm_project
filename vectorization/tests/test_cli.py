"""Test de extremo a extremo con TF-IDF y bloque de hora."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vectorization.cli import main


def run_tfidf(corpus_root: Path, out_dir: Path, *extra: str) -> int:
    """Lanza ``tfidf`` sobre el corpus sintético y devuelve el código."""
    return main(["tfidf", "--corpus-root", str(corpus_root),
                 "--output-dir", str(out_dir), "--min-df", "1",
                 "--svd", "3", "--count-emoji-tokens", *extra])


def test_tfidf_user_end_to_end(corpus_root: Path, tmp_path: Path) -> None:
    """X y metadata quedan alineados y X tiene filas unitarias."""
    out = tmp_path / "out"
    assert run_tfidf(corpus_root, out, "--unit", "user",
                     "--name", "prueba") == 0
    folder = out / "prueba"
    matrix = np.load(folder / "X.npy")
    metadata = pd.read_csv(folder / "metadata.csv")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert matrix.shape[0] == len(metadata) == 3
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    assert config["shape"] == list(matrix.shape)
    assert (folder / "vectorizer.joblib").exists()
    assert "vectorize" in config["timings_seconds"]


def test_time_block_changes_width_and_existing_folder_is_protected(
        corpus_root: Path, tmp_path: Path) -> None:
    """Con --time se añaden columnas; repetir sin --overwrite da error."""
    out = tmp_path / "out"
    assert run_tfidf(corpus_root, out, "--time", "0.2",
                     "--name", "con_hora") == 0
    matrix = np.load(out / "con_hora" / "X.npy")
    assert matrix.shape[1] == 3 + 4
    assert (out / "con_hora" / "blocks" / "time.npy").exists()
    assert run_tfidf(corpus_root, out, "--name", "con_hora") == 1
    assert run_tfidf(corpus_root, out, "--name", "con_hora",
                     "--overwrite") == 0
