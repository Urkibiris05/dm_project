"""
Módulo de clustering DBSCAN con selección automática de epsilon.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


def seleccionar_columna_clase(df, preferida: str = "label") -> str | None:
    """Devuelve la columna de evaluación solicitada si está disponible.

    ``label`` representa el riesgo anotado. ``disorder``/``trastorno`` se
    mantienen como alternativas explícitas para analizar el subcorpus.
    """
    candidatos = [preferida, "label", "disorder", "trastorno"]
    for columna in dict.fromkeys(candidatos):
        if columna in df.columns:
            return columna
    return None


def evaluate_clusters(
    X_proc: np.ndarray,
    clusters: np.ndarray,
    y_real: np.ndarray | None,
    metric: str = "euclidean"
) -> dict:
    """Evalúa la calidad de una partición de clusters."""
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = np.sum(clusters == -1)
    noise_ratio = n_noise / len(clusters)

    mask_valid = clusters >= 0
    if n_clusters >= 2 and mask_valid.sum() > n_clusters:
        sil = silhouette_score(X_proc[mask_valid], clusters[mask_valid], metric=metric)
        ari = adjusted_rand_score(y_real[mask_valid], clusters[mask_valid]) if y_real is not None else -1
        nmi = normalized_mutual_info_score(y_real[mask_valid], clusters[mask_valid]) if y_real is not None else -1
    else:
        sil, ari, nmi = -1, -1, -1

    return {
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "silhouette": sil,
        "ari": ari,
        "nmi": nmi
    }


def calculate_combined_score(sil: float, nmi: float, noise_ratio: float) -> float:
    """Calcula un score combinado para seleccionar la mejor partición."""
    return sil + nmi - (noise_ratio * 0.5)


def process_vectorization(
    nombre: str,
    info: dict,
    percentiles: list[int] = None,
    metric: str = "euclidean",
    max_noise_ratio: float = 0.45,
    min_clusters: int = 2,
    max_clusters: int = 30,
    label_column: str = "label",
) -> dict:
    """Ejecuta DBSCAN con barrido de percentiles para una vectorización."""
    if percentiles is None:
        percentiles = [50, 65, 75, 85, 92]

    X_proc = info["X_proc"]
    df = info["df"]
    k_distances = info["k_distances"]
    k = info["k"]

    print(f"\n{'='*75}")
    print(f"  Analizando: {nombre} ({info['info_red']})")
    print(f"{'='*75}")

    col_clase = seleccionar_columna_clase(df, label_column)
    y_real = df[col_clase].astype("category").cat.codes.values if col_clase else None

    mejor_resultado = None
    mejor_score = -10
    ultimo_resultado = None

    for percentil in percentiles:
        eps = float(np.percentile(k_distances, percentil))
        if eps <= 0:
            continue

        db = DBSCAN(eps=eps, min_samples=k, metric=metric, n_jobs=-1)
        clusters = db.fit_predict(X_proc)

        metricas = evaluate_clusters(X_proc, clusters, y_real, metric)

        print(
            f"  [P{percentil:2d} | eps={eps:.4f}] -> "
            f"Clusters: {metricas['n_clusters']:3d} | "
            f"Ruido: {metricas['noise_ratio']:5.1%} | "
            f"Sil: {metricas['silhouette']:6.3f} | "
            f"ARI: {metricas['ari']:6.3f} | "
            f"NMI: {metricas['nmi']:6.3f}"
        )

        ultimo_resultado = {
            "vectorizacion": nombre,
            "percentil": f"P{percentil}",
            "eps": eps,
            **metricas,
            "clusters": clusters,
            "X_2d": info["X_2d"],
            "df": df
        }

        if (min_clusters <= metricas['n_clusters'] <= max_clusters and
                metricas['noise_ratio'] <= max_noise_ratio):
            score = calculate_combined_score(
                metricas['silhouette'], metricas['nmi'], metricas['noise_ratio']
            )
            if score > mejor_score:
                mejor_score = score
                mejor_resultado = dict(ultimo_resultado)

    if mejor_resultado is None and ultimo_resultado is not None:
        mejor_resultado = dict(ultimo_resultado)
        mejor_resultado["percentil"] = f"P{percentil}*"

    return mejor_resultado


def show_global_summary(resultados: list[dict]) -> None:
    """Muestra una tabla resumen comparativa de todas las vectorizaciones."""
    print(f"\n\n{'='*95}")
    print("RESUMEN COMPARATIVO GLOBAL (UMAP + DBSCAN)")
    print(f"{'='*95}")
    print(
        f"{'Vectorización':<45} {'Perc':>5} {'Eps':>7} {'Clusters':>8} "
        f"{'Ruido':>7} {'Sil':>7} {'ARI':>7} {'NMI':>7}"
    )
    print("-" * 95)
    for r in resultados:
        nombre_corto = r["vectorizacion"][:43]
        sil_str = f"{r['silhouette']:.3f}" if r['silhouette'] >= 0 else "  -"
        ari_str = f"{r['ari']:.3f}" if r['ari'] >= 0 else "  -"
        nmi_str = f"{r['nmi']:.3f}" if r['nmi'] >= 0 else "  -"
        print(
            f"{nombre_corto:<45} {r['percentil']:>5} {r['eps']:>7.4f} "
            f"{r['n_clusters']:>8} {r['noise_ratio']:>6.1%} "
            f"{sil_str:>7} {ari_str:>7} {nmi_str:>7}"
        )


def apply_clustering(datos_procesados: dict, **kwargs) -> list[dict]:
    """Ejecuta el pipeline completo de clustering DBSCAN para todas las vectorizaciones."""
    resultados = []
    for nombre, info in datos_procesados.items():
        resultado = process_vectorization(nombre, info, **kwargs)
        resultados.append(resultado)

    show_global_summary(resultados)

    return resultados
