"""Combinación ponderada de bloques de rasgos (aportación P3).

``x = [(1 - α_afecto - α_hora)·e/‖e‖ ; α_afecto·p/‖p‖ ; α_hora·t/‖t‖]``

con ``e`` el bloque semántico (TF-IDF o embedding), ``p`` las probabilidades
de pysentimiento y ``t`` los rasgos de hora. Cada bloque se normaliza por
fila antes de pesarlo, de modo que el peso controla cuánto mueve cada bloque
las distancias.
"""

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Divide cada fila por su norma L2 (las filas nulas se dejan a cero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def standardize(matrix: np.ndarray) -> np.ndarray:
    """Resta la media y divide por la desviación de cada columna.

    Las columnas constantes quedan a cero. Se usa en el bloque de hora, cuyas
    columnas tienen escalas muy distintas.
    """
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    return (matrix - matrix.mean(axis=0)) / std


def combine_blocks(
    semantic: np.ndarray | None,
    affect: np.ndarray | None,
    time_block: np.ndarray | None,
    affect_weight: float,
    time_weight: float,
) -> np.ndarray:
    """Concatena los bloques presentes con sus pesos.

    Con bloque semántico, su peso es ``1 - afecto - hora``. Sin él (subcomando
    ``features``) los pesos de afecto y hora se reescalan para sumar 1.
    Devuelve una matriz ``float32``.
    """
    weights: list[float] = []
    parts: list[np.ndarray] = []
    if affect is not None:
        weights.append(affect_weight)
        parts.append(l2_normalize(affect))
    if time_block is not None:
        weights.append(time_weight)
        parts.append(l2_normalize(standardize(time_block)))
    if semantic is not None:
        weights.insert(0, 1.0 - sum(weights))
        parts.insert(0, l2_normalize(semantic))
    else:
        total = sum(weights)
        weights = [weight / total for weight in weights]
    blocks = [weight * part for weight, part in zip(weights, parts)]
    return np.hstack(blocks).astype(np.float32)
