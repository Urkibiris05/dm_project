"""Barrido reproducible de DBSCAN sobre representaciones de usuario.

E2 compara MiniLM y E5. La comparación de ``concat``, ``mean``, ``max`` y
``blocks`` constituye E3. La selección usa DBCV, cobertura, ruido y
estabilidad; las etiquetas solo se calculan como evaluación final.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.sparse.csgraph import minimum_spanning_tree


ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = ROOT / "vectorizations2"
RESULT_DIR = Path(__file__).resolve().parent / "resultados"
REPRESENTATIONS = tuple(
    sorted(
        path.name
        for path in VECTOR_DIR.glob("user__all__*")
        if (path / "X.npy").exists()
        and (path / "metadata.csv").exists()
        and np.load(path / "X.npy", mmap_mode="r").shape[0] == 820
    )
)
MIN_SAMPLES_VALUES = (5, 10, 20, 40)
PERCENTILES = (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90)
SPACES = ("original", "centered", "pca32", "pca64", "pca128", "pca256")
MAX_NOISE_RATIO = 0.40
MAX_LARGEST_CLUSTER_FRAC = 0.70
STABILITY_REPEATS = 20
STABILITY_SEED = 20261018


def normalize_rows(X: np.ndarray) -> np.ndarray:
    """Normaliza filas para conservar la geometría coseno."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def transform_space(X: np.ndarray, space: str) -> np.ndarray:
    """Construye el espacio experimental y renormaliza sus filas."""
    if space == "original":
        return normalize_rows(X)
    if space == "centered":
        return normalize_rows(X - X.mean(axis=0, keepdims=True))
    if space.startswith("pca"):
        components = int(space[3:])
        reduced = PCA(n_components=components, random_state=STABILITY_SEED).fit_transform(X)
        return normalize_rows(reduced)
    raise ValueError(f"Espacio desconocido: {space}")


def spaces_for(X: np.ndarray) -> tuple[str, ...]:
    """Devuelve espacios PCA válidos para la dimensionalidad de X."""
    max_components = min(X.shape)
    pca_spaces = tuple(
        f"pca{components}"
        for components in (32, 64, 128, 256)
        if components <= max_components
    )
    return ("original", "centered", *pca_spaces)


def dbcv_score(
    distances: np.ndarray, labels: np.ndarray, min_samples: int
) -> float:
    """Calcula una aproximación reproducible de DBCV para clusters de densidad.

    Usa distancias de alcanzabilidad mutua, la dispersión del árbol de
    expansión mínimo de cada cluster y la separación entre clusters. El ruido
    no participa en la puntuación, pero se informa por separado.
    """
    clusters = [c for c in np.unique(labels) if c >= 0]
    if len(clusters) < 2:
        return float("nan")
    core_distances = np.sort(distances, axis=1)[:, min_samples - 1]
    mutual = np.maximum(
        distances,
        np.maximum(core_distances[:, None], core_distances[None, :]),
    )
    weighted_sum = 0.0
    total_points = 0
    for cluster in clusters:
        members = np.flatnonzero(labels == cluster)
        if len(members) == 1:
            sparseness = 0.0
        else:
            tree = minimum_spanning_tree(mutual[np.ix_(members, members)])
            sparseness = float(tree.toarray().max())
        other = np.flatnonzero((labels >= 0) & (labels != cluster))
        separation = float(mutual[np.ix_(members, other)].min())
        denominator = max(separation, sparseness)
        validity = (
            (separation - sparseness) / denominator
            if denominator > 0 else 0.0
        )
        weighted_sum += len(members) * validity
        total_points += len(members)
    return weighted_sum / total_points if total_points else float("nan")


def stability_score(
    X: np.ndarray,
    reference: np.ndarray,
    eps: float,
    min_samples: int,
) -> tuple[float, float]:
    """Mide estabilidad con bootstrap y perturbaciones de epsilon."""
    rng = np.random.default_rng(STABILITY_SEED)
    scores = []
    for _ in range(STABILITY_REPEATS):
        indices = rng.integers(0, len(X), len(X))
        sample = X[indices]
        factor = rng.uniform(0.95, 1.05)
        labels = DBSCAN(
            eps=eps * factor,
            min_samples=min_samples,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ).fit_predict(sample)
        scores.append(adjusted_rand_score(reference[indices], labels))
    return float(np.mean(scores)), float(np.std(scores))


def evaluate(
    X: np.ndarray,
    metadata: pd.DataFrame,
    min_samples: int,
    percentile: int,
    eps: float,
    space: str,
    distances: np.ndarray,
    calculate_dbcv: bool = False,
    calculate_stability: bool = False,
) -> dict[str, float | int | str]:
    """Ejecuta una configuración y devuelve métricas internas y externas."""
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    ).fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = float(np.mean(labels == -1))
    cluster_sizes = np.bincount(labels[labels >= 0])
    largest_cluster_frac = (
        float(cluster_sizes.max() / len(labels)) if len(cluster_sizes) else 0.0
    )
    valid = labels >= 0
    if n_clusters >= 2 and valid.sum() > n_clusters:
        silhouette = float(
            silhouette_score(X[valid], labels[valid], metric="cosine")
        )
    else:
        silhouette = float("nan")

    row: dict[str, float | int | str] = {
        "min_samples": min_samples,
        "percentile": percentile,
        "eps": eps,
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "largest_cluster_frac": largest_cluster_frac,
        "dbcv": (
            dbcv_score(distances, labels, min_samples)
            if calculate_dbcv else float("nan")
        ),
        "silhouette": silhouette,
        "space": space,
    }
    if calculate_stability:
        mean, std = stability_score(X, labels, eps, min_samples)
        row["stab_ari_mean"] = mean
        row["stab_ari_std"] = std
    for column in ("label", "disorder"):
        target = metadata[column].astype("category").cat.codes.to_numpy()
        row[f"{column}_ari"] = float(adjusted_rand_score(target, labels))
        row[f"{column}_nmi"] = float(
            normalized_mutual_info_score(target, labels)
        )
    return row


def run_representation(name: str) -> list[dict[str, float | int | str]]:
    """Explora una representación con todos los min_samples y percentiles."""
    folder = VECTOR_DIR / name
    raw_X = np.load(folder / "X.npy")
    metadata = pd.read_csv(folder / "metadata.csv")
    rows = []
    for space in spaces_for(raw_X):
        X = transform_space(raw_X, space)
        distances = pairwise_distances(X, metric="cosine")
        for min_samples in MIN_SAMPLES_VALUES:
            neighbor_distances, _ = NearestNeighbors(
                n_neighbors=min_samples, metric="cosine", algorithm="brute"
            ).fit(X).kneighbors(X)
            k_distances = np.sort(neighbor_distances[:, -1])
            for percentile in PERCENTILES:
                eps = float(np.percentile(k_distances, percentile))
                row = evaluate(
                    X, metadata, min_samples, percentile, eps, space,
                    distances, calculate_dbcv=True,
                )
                row["representation"] = name
                rows.append(row)
    return rows


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in REPRESENTATIONS:
        print(f"Procesando {name}...")
        rows.extend(run_representation(name))
    columns = [
        "representation",
        "space",
        "min_samples",
        "percentile",
        "eps",
        "n_clusters",
        "noise_ratio",
        "largest_cluster_frac",
        "dbcv",
        "silhouette",
        "label_ari",
        "label_nmi",
        "disorder_ari",
        "disorder_nmi",
    ]
    results = pd.DataFrame(rows)[columns].sort_values(
        ["representation", "space", "min_samples", "percentile"]
    )
    results.to_csv(RESULT_DIR / "e2_e3_dbscan_sweep.csv", index=False)

    eligible = results[
        (results["n_clusters"] >= 2)
        & (results["noise_ratio"] <= MAX_NOISE_RATIO)
        & (results["largest_cluster_frac"] < MAX_LARGEST_CLUSTER_FRAC)
        & (results["dbcv"] > 0)
    ].copy()
    shortlist = (
        eligible.sort_values("dbcv", ascending=False)
        .groupby(["representation", "space"], as_index=False)
        .head(5)
    )
    stable_rows = []
    for _, candidate in shortlist.iterrows():
        folder = VECTOR_DIR / candidate["representation"]
        raw_X = np.load(folder / "X.npy")
        X = transform_space(raw_X, candidate["space"])
        evaluated = evaluate(
            X,
            pd.read_csv(folder / "metadata.csv"),
            int(candidate["min_samples"]),
            int(candidate["percentile"]),
            float(candidate["eps"]),
            str(candidate["space"]),
            pairwise_distances(X, metric="cosine"),
            calculate_dbcv=True,
            calculate_stability=True,
        )
        evaluated["representation"] = candidate["representation"]
        stable_rows.append(evaluated)
    stable = pd.DataFrame(stable_rows)
    if not stable.empty:
        stable["valid_stable"] = stable["stab_ari_mean"] >= 0.80
        summary = (
            stable.sort_values(
                ["representation", "space", "valid_stable", "dbcv"],
                ascending=[True, True, False, False],
            )
            .groupby(["representation", "space"], as_index=False)
            .first()
        )
    else:
        summary = pd.DataFrame(
            columns=columns + ["stab_ari_mean", "stab_ari_std", "valid_stable"]
        )
    summary.to_csv(RESULT_DIR / "e2_e3_internal_summary.csv", index=False)
    stable.to_csv(RESULT_DIR / "e2_e3_stability_shortlist.csv", index=False)
    print(f"Resultados guardados en {RESULT_DIR}")


if __name__ == "__main__":
    main()
