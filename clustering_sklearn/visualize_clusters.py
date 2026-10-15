"""
Módulo de visualización de clusters DBSCAN sobre proyección UMAP.
"""

import matplotlib.pyplot as plt
import numpy as np


def graficar_clusters_2d(
    resultados: list[dict],
    figsize: tuple[int, int] = (15, 12),
    mostrar: bool = True,
    guardar_path: str = None
) -> None:
    """Grafica los clusters DBSCAN en subplots 2x2 sobre la proyección UMAP 2D."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, r in enumerate(resultados):
        ax = axes[idx]
        X_2d = r["X_2d"]
        clusters = r["clusters"]
        ruido = clusters == -1

        ax.scatter(
            X_2d[ruido, 0], X_2d[ruido, 1],
            c="lightgrey", s=12, alpha=0.4, label="Ruido (-1)"
        )

        if (~ruido).sum() > 0:
            ax.scatter(
                X_2d[~ruido, 0], X_2d[~ruido, 1],
                c=clusters[~ruido], cmap="tab20", s=20, alpha=0.85
            )

        ax.set_title(
            f"{r['vectorizacion']} ({r['percentil']})\n"
            f"Clusters: {r['n_clusters']} | Ruido: {r['noise_ratio']:.1%} | "
            f"Sil: {r['silhouette']:.3f} | NMI: {r['nmi']:.3f}",
            fontsize=10
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("Proyección UMAP 2D de los Clusters con DBSCAN", fontsize=14, y=1.01)
    plt.tight_layout()

    if guardar_path:
        plt.savefig(guardar_path, dpi=150, bbox_inches="tight")
        print(f"Gráfica guardada en: {guardar_path}")

    if mostrar:
        plt.show()
    else:
        plt.close()
