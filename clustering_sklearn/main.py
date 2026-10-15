"""
Pipeline principal: UMAP + DBSCAN + Visualización.

Este script ejecuta el pipeline completo de análisis de clusters:
1. Reducción de dimensionalidad con UMAP para todas las vectorizaciones
2. Clustering DBSCAN con selección automática de epsilon
3. Visualización de los resultados

Uso:
    python main.py
"""

from pathlib import Path

from vectorization.timing import PhaseTimer

from clustering_sklearn.umap_reduction import apply_reduction
from clustering_sklearn.dbscan_clustering import apply_clustering
from clustering_sklearn.visualize_clusters import graficar_clusters_2d


# Configuración
BASE_DIR = Path("./vectorizations")
MIN_SAMPLES = 15
METRIC = "euclidean"
ALGORITHM = "brute"  # "brute" o "blocked" (implementación propia)


def main():
    print("=" * 75)
    print("  PIPELINE DE CLUSTERING: UMAP + DBSCAN")
    print("=" * 75)
    print(f"  Directorio base: {BASE_DIR}")
    print(f"  Min samples (k): {MIN_SAMPLES}")
    print(f"  Métrica: {METRIC}")
    print(f"  Algoritmo: {ALGORITHM}")
    print("=" * 75)

    timer = PhaseTimer()

    # Paso 1: Reducción de dimensionalidad con UMAP
    print("\n" + "=" * 75)
    print("  PASO 1: REDUCCIÓN UMAP")
    print("=" * 75)
    datos_procesados = apply_reduction(BASE_DIR, min_samples=MIN_SAMPLES, timer=timer)

    # Paso 2: Clustering con DBSCAN
    print("\n" + "=" * 75)
    print("  PASO 2: CLUSTERING DBSCAN")
    print("=" * 75)
    resultados = apply_clustering(datos_procesados, metric=METRIC, algorithm=ALGORITHM)

    # Paso 3: Visualización de resultados
    print("\n" + "=" * 75)
    print("  PASO 3: VISUALIZACIÓN")
    print("=" * 75)
    graficar_clusters_2d(resultados, guardar_path="clusters_dbscan.png")

    # Mostrar tiempos
    print("\n" + "=" * 75)
    print("  TIEMPOS POR FASE")
    print("=" * 75)
    for phase, seconds in timer.seconds.items():
        print(f"  {phase}: {seconds:.3f}s")

    print("\n" + "=" * 75)
    print("  PIPELINE COMPLETADO")
    print("=" * 75)


if __name__ == "__main__":
    main()
