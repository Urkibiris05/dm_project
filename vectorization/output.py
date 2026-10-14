"""Escritura de una vectorización en su carpeta de salida.

Contenido de ``vectorizations/<nombre>/``:

- ``X.npy``: matriz float32, una fila por unidad.
- ``metadata.csv``: una fila por fila de X (identificadores, etiquetas, texto).
- ``config.json``: argumentos, forma, tiempos por fase e informe de pasos.
- ``vectorizer.joblib``: TF-IDF+SVD o PCA ajustados (si los hay).
- ``blocks/*.npy``: bloques sueltos, para rehacer la combinación con otros
  pesos sin volver a vectorizar.
"""

import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# GitHub rechaza archivos de más de 100 MB.
SIZE_WARNING_BYTES: int = 95_000_000


def prepare_directory(out_dir: Path, overwrite: bool) -> None:
    """Comprueba que la carpeta de salida se puede escribir.

    Falla si ya existe y no se pidió ``--overwrite``.
    """
    if out_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{out_dir} ya existe; usa --overwrite para reemplazarla")


def save_vectorization(
    out_dir: Path,
    matrix: np.ndarray,
    metadata: pd.DataFrame,
    config: dict,
    vectorizer: object | None,
    blocks: dict[str, np.ndarray],
) -> None:
    """Guarda la vectorización completa en ``out_dir``.

    Reemplaza el contenido anterior de la carpeta solo en este momento, una
    vez calculado todo, para que un fallo a mitad no destruya el resultado
    previo. Avisa por stderr si ``X.npy`` pasa de 95 MB.
    """
    if len(metadata) != len(matrix):
        raise ValueError("X y metadata deben tener el mismo número de filas")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    np.save(out_dir / "X.npy", matrix)
    metadata.to_csv(out_dir / "metadata.csv", index=False, encoding="utf-8")
    if vectorizer is not None:
        joblib.dump(vectorizer, out_dir / "vectorizer.joblib")
    if blocks:
        (out_dir / "blocks").mkdir()
        for name, block in blocks.items():
            np.save(out_dir / "blocks" / f"{name}.npy", block)
    config["shape"] = list(matrix.shape)
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    if matrix.nbytes > SIZE_WARNING_BYTES:
        print(f"AVISO: X.npy ocupa {matrix.nbytes / 1e6:.0f} MB; GitHub "
              "rechaza archivos de más de 100 MB (usa Git LFS o --pca).",
              file=sys.stderr)
