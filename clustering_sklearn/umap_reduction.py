"""
Módulo de reducción de dimensionalidad con UMAP.

Procesa todas las vectorizaciones encontradas en subcarpetas del directorio base,
aplicando StandardScaler y UMAP, guardando las representaciones 2D y calculando
las k-distancias necesarias para el posterior clustering DBSCAN.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import umap

from vectorization.timing import PhaseTimer


def detect_vectorizations(base_dir: Path) -> list[str]:
    """Detecta automáticamente todas las carpetas con un archivo X.npy."""
    return sorted([
        d.name for d in base_dir.iterdir()
        if d.is_dir() and (d / "X.npy").exists()
    ])


def load_config(carpeta: Path) -> dict:
    """Carga el archivo config.json de una carpeta de vectorización."""
    import json
    config_path = carpeta / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def process_vectorization(
    carpeta: Path,
    min_samples: int = 15,
    random_state: int = 42,
    timer: PhaseTimer = None
) -> dict:
    """Procesa una vectorización aplicando UMAP según el tipo de datos."""
    nombre = carpeta.name

    def _do_process():
        print(f"\n{'='*75}")
        print(f"  Procesando: {nombre}")
        print(f"{'='*75}")

        X = np.load(carpeta / "X.npy")
        df = pd.read_csv(carpeta / "metadata.csv") if (carpeta / "metadata.csv").exists() else pd.DataFrame()
        config = load_config(carpeta)

        print(f"  Dimensiones originales: {X.shape}")

        # Distinguimos si es una vectorización de texto (embeddings) o de características manuales (features)
        es_features_tabulares = "features" in nombre or X.shape[1] <= 20

        if es_features_tabulares:
            # Para variables tabulares (afecto/tiempo) usar Scaler y usar euclidea
            scaler = StandardScaler()
            X_proc = scaler.fit_transform(X)
            umap_metric = "euclidean"
            print("  [INFO] Aplicando StandardScaler y métrica euclídea (datos tabulares).")
        else:
            # Para embeddings de texto, no usar Scaler y usar coseno
            X_proc = X
            umap_metric = "cosine"
            print("  [INFO] Usando embeddings en bruto con métrica coseno.")


        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min_samples,
            min_dist=0.0, 
            metric=umap_metric,
            random_state=random_state
        )
        X_2d = reducer.fit_transform(X_proc)

        k = min_samples
        neigh = NearestNeighbors(n_neighbors=k, metric="euclidean" if not es_features_tabulares else "euclidean")
        neigh.fit(X_2d)
        distances, _ = neigh.kneighbors(X_2d)
        k_distances = np.sort(distances[:, k - 1])

        return {
            "X_proc": X_2d, 
            "X_2d": X_2d,
            "df": df,
            "k_distances": k_distances,
            "k": k,
            "info_red": f"{X.shape[1]}D -> UMAP ({umap_metric})",
            "config": config
        }

    if timer:
        with timer.phase(f"umap_{nombre}"):
            return _do_process()
    return _do_process()


def apply_reduction(base_dir: Path, min_samples: int = 15, timer: PhaseTimer = None) -> dict:
    """Ejecuta el pipeline completo de reducción UMAP para todas las vectorizaciones."""
    a_probar = detect_vectorizations(base_dir)
    print(f"Total de vectorizaciones encontradas para procesar: {len(a_probar)}\n")

    datos_procesados = {}
    for nombre in a_probar:
        carpeta = base_dir / nombre
        datos_procesados[nombre] = process_vectorization(
            carpeta, min_samples=min_samples, timer=timer
        )

    print(f"\n{'='*75}")
    print(f"  Reducción UMAP completada para {len(datos_procesados)} vectorizaciones")
    print(f"{'='*75}\n")

    return datos_procesados
