"""Medición del tiempo de cada fase del pipeline."""

import time
from collections.abc import Iterator
from contextlib import contextmanager


class PhaseTimer:
    """Guarda los segundos que tarda cada fase con nombre.

    Uso: ``with timer.phase("vectorize"): ...``. El resultado queda en
    ``timer.seconds`` y se escribe en ``config.json``.
    """

    def __init__(self) -> None:
        """Crea el temporizador vacío."""
        self.seconds: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Mide el bloque ``with`` y lo anota bajo ``name``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.seconds[name] = round(elapsed, 3)
