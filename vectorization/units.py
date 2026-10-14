"""Unidad de análisis: un vector por mensaje o un vector por usuario.

Todos los mensajes de un usuario están contiguos en el DataFrame (lo
garantiza ``corpus.load_corpus`` y el preprocesado lo respeta). Gracias a eso,
``owner`` (la unidad a la que pertenece cada fila) es no decreciente y las
agregaciones se hacen con ``reduceat``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

USER_COLUMNS: list[str] = [
    "subject_key", "disorder", "split", "subject", "label", "target",
]
MESSAGE_COLUMNS: list[str] = USER_COLUMNS + ["id_message", "date"]


@dataclass
class Units:
    """Unidades de análisis ya construidas.

    ``texts``: texto de cada unidad (el mensaje, o los mensajes del usuario
    concatenados por orden de fecha). ``metadata``: una fila por unidad con
    sus identificadores, etiquetas y texto. ``owner``: para cada mensaje del
    DataFrame, índice de la unidad a la que pertenece.
    """

    texts: list[str]
    metadata: pd.DataFrame
    owner: np.ndarray


def build_units(df: pd.DataFrame, unit: str) -> Units:
    """Construye las unidades de análisis a partir de los mensajes.

    Con ``unit="message"`` hay una unidad por mensaje. Con ``unit="user"``
    hay una por usuario, cuyo texto es la concatenación de sus mensajes.
    """
    df = df.reset_index(drop=True)
    if unit == "message":
        metadata = df[MESSAGE_COLUMNS].copy()
        metadata["text"] = df["message"]
        return Units(df["message"].tolist(), metadata, np.arange(len(df)))

    rows = []
    texts = []
    for _, group in df.groupby("subject_key", sort=False):
        first = group.iloc[0]
        row = {column: first[column] for column in USER_COLUMNS}
        row["n_messages"] = len(group)
        rows.append(row)
        texts.append(" ".join(group["message"]))
    metadata = pd.DataFrame(rows)
    metadata["text"] = texts
    owner = pd.factorize(df["subject_key"])[0]
    return Units(texts, metadata, owner)


def aggregate_rows(
    matrix: np.ndarray,
    owner: np.ndarray,
    how: str,
) -> np.ndarray:
    """Agrupa las filas de ``matrix`` por unidad (``mean`` o ``max``).

    ``owner[i]`` es la unidad de la fila ``i``; debe ser no decreciente
    (filas de una misma unidad contiguas). Devuelve una fila por unidad.
    """
    if len(owner) == 0:
        raise ValueError("No hay filas que agregar")
    if np.any(np.diff(owner) < 0):
        raise ValueError("Las filas de cada unidad deben ser contiguas")
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    if how == "mean":
        counts = np.diff(np.r_[starts, len(owner)])
        return np.add.reduceat(matrix, starts, axis=0) / counts[:, None]
    if how == "max":
        return np.maximum.reduceat(matrix, starts, axis=0)
    raise ValueError(f"Agregación desconocida: {how}")


def split_blocks(text: str, block_words: int) -> list[str]:
    """Parte un texto en bloques de ``block_words`` palabras.

    Si el último bloque es menor que la mitad del tamaño se une al
    anterior, para que un resto diminuto no pese como un bloque entero al
    hacer la media. Un texto vacío da un único bloque vacío.
    """
    words = text.split()
    blocks = [words[i:i + block_words]
              for i in range(0, len(words), block_words)]
    if len(blocks) > 1 and len(blocks[-1]) < block_words // 2:
        remainder = blocks.pop()
        blocks[-1] = blocks[-1] + remainder
    return [" ".join(block) for block in blocks] or [""]


def make_blocks(
    texts: list[str],
    block_words: int,
) -> tuple[list[str], np.ndarray]:
    """Parte el texto de cada unidad en bloques.

    Devuelve ``(bloques, owner)``, donde ``owner[j]`` es la unidad de la que
    sale el bloque ``j``.
    """
    blocks: list[str] = []
    owner: list[int] = []
    for unit_index, text in enumerate(texts):
        pieces = split_blocks(text, block_words)
        blocks.extend(pieces)
        owner.extend([unit_index] * len(pieces))
    return blocks, np.array(owner)
