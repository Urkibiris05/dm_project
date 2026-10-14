"""Rasgos afectivos de cada mensaje con pysentimiento (español).

Cada tarea devuelve un vector de probabilidades por mensaje. Se guardan en
una caché en disco por tarea para no repetir el cálculo.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .cache import VectorCache

# Etiquetas de cada tarea (orden alfabético, fijo para que las columnas sean
# siempre las mismas).
AFFECT_LABELS: dict[str, list[str]] = {
    "sentiment": ["NEG", "NEU", "POS"],
    "emotion": ["anger", "disgust", "fear", "joy", "others", "sadness",
                "surprise"],
    "hate_speech": ["aggressive", "hateful", "targeted"],
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Añade las opciones del bloque afectivo (comunes a los subcomandos)."""
    parser.add_argument("--affect", type=float, default=0.0, metavar="W",
                        help="peso del bloque de pysentimiento (0 = no se "
                        "usa)")
    parser.add_argument("--affect-tasks", nargs="+", default=["sentiment",
                                                              "emotion"],
                        choices=sorted(AFFECT_LABELS),
                        help="tareas de pysentimiento (def. sentiment "
                        "emotion)")


def affect_columns(tasks: list[str]) -> list[str]:
    """Devuelve los nombres de columna ``tarea:etiqueta`` de las tareas."""
    return [f"{task}:{label}" for task in tasks
            for label in AFFECT_LABELS[task]]


def make_task_encoder(
    task: str,
    batch_size: int,
) -> Callable[[list[str]], np.ndarray]:
    """Crea una función ``textos -> probabilidades`` para una tarea.

    El analizador de pysentimiento se carga en la primera llamada, así que
    si todo está en la caché no se importa la librería.
    """
    state: dict = {}

    def encode(texts: list[str]) -> np.ndarray:
        """Calcula las probabilidades de cada texto para la tarea."""
        if "analyzer" not in state:
            try:
                from pysentimiento import create_analyzer
            except ImportError as error:
                raise ImportError(
                    "Falta pysentimiento: pip install -r "
                    "vectorization/requirements.txt") from error
            state["analyzer"] = create_analyzer(task=task, lang="es")
        labels = AFFECT_LABELS[task]
        rows = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            outputs = state["analyzer"].predict(batch)
            rows.extend([[output.probas[label] for label in labels]
                         for output in outputs])
        return np.array(rows, dtype=np.float32)

    return encode


def message_affect(
    texts: list[str],
    tasks: list[str],
    cache_dir: Path,
    use_cache: bool,
    batch_size: int = 64,
) -> tuple[np.ndarray, dict]:
    """Calcula las probabilidades de pysentimiento de cada mensaje.

    Concatena por columnas las tareas pedidas (3 + 7 + 3 dimensiones como
    máximo). Devuelve ``(matriz, info)`` con los aciertos de caché.
    """
    parts = []
    info = {"cache_hits": 0, "cache_misses": 0}
    for task in tasks:
        cache = VectorCache(cache_dir / "affect" / f"{task}.npz",
                            enabled=use_cache)
        encode = make_task_encoder(task, batch_size)
        parts.append(cache.get_matrix(texts, encode))
        info["cache_hits"] += cache.hits
        info["cache_misses"] += cache.misses
    return np.hstack(parts), info
