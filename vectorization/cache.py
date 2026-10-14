"""Caché en disco de vectores calculados por texto.

Se guarda un archivo ``.npz`` por modelo con los hashes de los textos y sus
vectores. Como la clave es el texto, la caché sirve para cualquier subconjunto
del corpus y para cualquier ejecución que repita mensajes.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np


def text_key(text: str) -> str:
    """Devuelve el hash SHA-1 de un texto, usado como clave de caché."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class VectorCache:
    """Caché persistente ``texto -> vector``.

    Con ``enabled=False`` ignora el archivo existente y no escribe nada
    (opción ``--no-cache``). ``hits`` y ``misses`` cuentan textos únicos
    encontrados y calculados, para dejarlo anotado en ``config.json``.
    """

    def __init__(self, path: Path, enabled: bool = True) -> None:
        """Abre la caché y carga el archivo si existe y está habilitada."""
        self.path = Path(path)
        self.enabled = enabled
        self.keys: list[str] = []
        self.index: dict[str, int] = {}
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self.hits = 0
        self.misses = 0
        if enabled and self.path.exists():
            with np.load(self.path) as data:
                self.keys = data["keys"].tolist()
                self.vectors = data["vectors"]
            self.index = {key: i for i, key in enumerate(self.keys)}

    def save(self) -> None:
        """Escribe la caché en disco (a un temporal y luego lo renombra)."""
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(temporary, keys=np.array(self.keys), vectors=self.vectors)
        temporary.replace(self.path)

    def get_matrix(
        self,
        texts: list[str],
        compute: Callable[[list[str]], np.ndarray],
        chunk_size: int = 2048,
    ) -> np.ndarray:
        """Devuelve una fila por texto, calculando solo los que faltan.

        ``compute`` recibe una lista de textos y devuelve su matriz de
        vectores. Los textos nuevos se calculan por trozos y la caché se
        guarda tras cada trozo, de modo que una interrupción no pierde todo.
        """
        keys = [text_key(text) for text in texts]
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts):
            if key not in self.index and key not in missing:
                missing[key] = text
        self.hits += len(set(keys)) - len(missing)
        self.misses += len(missing)

        pending = list(missing.items())
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start:start + chunk_size]
            new_vectors = np.asarray(
                compute([text for _, text in chunk]), dtype=np.float32)
            if self.vectors.size == 0:
                self.vectors = new_vectors
            else:
                self.vectors = np.vstack([self.vectors, new_vectors])
            for key, _ in chunk:
                self.index[key] = len(self.keys)
                self.keys.append(key)
            self.save()
        rows = [self.index[key] for key in keys]
        return self.vectors[rows]
