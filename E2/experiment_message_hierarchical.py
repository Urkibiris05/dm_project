"""DBSCAN jerárquico: mensajes primero y sujetos después.

La primera etapa agrupa mensajes individuales. La segunda representa cada
sujeto mediante la distribución normalizada de sus mensajes entre los
clústeres de la primera etapa, incluyendo una columna para el ruido, y vuelve
a aplicar DBSCAN sobre esas distribuciones.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = ROOT / "vectorizations2"
RESULT_DIR = Path(__file__).resolve().parent / "resultados"
MIN_SAMPLES = (5, 10, 20, 40)
USER_MIN_SAMPLES = (5, 10, 20, 40)
PERCENTILES = (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90)
MAX_NOISE_RATIO = 0.40
MAX_LARGEST_CLUSTER_FRAC = 0.70


def positive_eps(value: float) -> float:
    """Devuelve el menor epsilon positivo cuando el percentil es cero."""
    return max(float(value), float(np.nextafter(0.0, 1.0)))


def cluster_stats(labels: np.ndarray) -> tuple[int, float, float]:
    """Devuelve número de clústeres, ruido y tamaño relativo máximo."""
    clusters = labels >= 0
    n_clusters = len(set(labels)) - int(-1 in labels)
    noise_ratio = float(np.mean(~clusters))
    sizes = np.bincount(labels[clusters])
    largest = float(sizes.max() / len(labels)) if len(sizes) else 0.0
    return n_clusters, noise_ratio, largest


def user_profiles(
    metadata: pd.DataFrame, message_labels: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """Construye perfiles normalizados de clústeres de mensajes por sujeto."""
    subject_codes, subjects = pd.factorize(metadata["subject_key"], sort=False)
    categories = np.unique(message_labels)
    category_to_column = {int(value): index for index, value in enumerate(categories)}
    profiles = np.zeros((len(subjects), len(categories)), dtype=np.float32)
    np.add.at(
        profiles,
        (subject_codes, np.array([category_to_column[int(x)] for x in message_labels])),
        1,
    )
    profiles /= np.maximum(profiles.sum(axis=1, keepdims=True), 1)
    user_metadata = (
        metadata.drop_duplicates("subject_key")
        .set_index("subject_key")
        .reindex(subjects)
        .rename_axis("subject_key")
        .reset_index()
    )
    return user_metadata, profiles


def run_representation(
    name: str, vector_dir: Path, save_message_assignments: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ejecuta las dos etapas para una representación de mensajes."""
    folder = vector_dir / name
    X = np.load(folder / "X.npy", mmap_mode="r")
    metadata = pd.read_csv(folder / "metadata.csv")
    message_rows: list[dict[str, object]] = []
    user_rows: list[dict[str, object]] = []
    assignments: list[pd.DataFrame] = []
    message_assignments: list[pd.DataFrame] = []

    for min_samples in MIN_SAMPLES:
        distances, _ = NearestNeighbors(
            n_neighbors=min_samples, metric="cosine", algorithm="brute", n_jobs=-1
        ).fit(X).kneighbors(X)
        kdist = np.sort(distances[:, -1])
        for percentile in PERCENTILES:
            eps_message = positive_eps(np.percentile(kdist, percentile))
            message_labels = DBSCAN(
                eps=eps_message,
                min_samples=min_samples,
                metric="cosine",
                algorithm="brute",
                n_jobs=-1,
            ).fit_predict(X)
            n_message_clusters, message_noise, largest_message = cluster_stats(message_labels)
            message_rows.append({
                "representation": name,
                "min_samples_message": min_samples,
                "percentile_message": percentile,
                "eps_message": eps_message,
                "n_message_clusters": n_message_clusters,
                "message_noise_ratio": message_noise,
                "largest_message_cluster_frac": largest_message,
            })
            if save_message_assignments:
                message_assignments.append(pd.DataFrame({
                    "representation": name,
                    "min_samples_message": min_samples,
                    "percentile_message": percentile,
                    "eps_message": eps_message,
                    "subject_key": metadata["subject_key"],
                    "id_message": metadata["id_message"],
                    "message_cluster": message_labels,
                }))

            user_metadata, profiles = user_profiles(metadata, message_labels)
            for user_min_samples in USER_MIN_SAMPLES:
                if user_min_samples > len(profiles):
                    continue
                user_distances, _ = NearestNeighbors(
                    n_neighbors=user_min_samples,
                    metric="cosine",
                    algorithm="brute",
                    n_jobs=-1,
                ).fit(profiles).kneighbors(profiles)
                user_kdist = np.sort(user_distances[:, -1])
                for user_percentile in PERCENTILES:
                    eps_user = positive_eps(
                        np.percentile(user_kdist, user_percentile)
                    )
                    user_labels = DBSCAN(
                        eps=eps_user,
                        min_samples=user_min_samples,
                        metric="cosine",
                        algorithm="brute",
                        n_jobs=-1,
                    ).fit_predict(profiles)
                    n_user_clusters, user_noise, largest_user = cluster_stats(user_labels)
                    valid = user_labels >= 0
                    if n_user_clusters >= 2 and valid.sum() > n_user_clusters:
                        silhouette = float(
                            silhouette_score(
                                profiles[valid], user_labels[valid], metric="cosine"
                            )
                        )
                    else:
                        silhouette = float("nan")
                    user_rows.append({
                    "representation": name,
                    "min_samples_message": min_samples,
                    "percentile_message": percentile,
                    "eps_message": eps_message,
                    "n_message_clusters": n_message_clusters,
                    "message_noise_ratio": message_noise,
                    "user_percentile": user_percentile,
                    "eps_user": eps_user,
                    "min_samples_user": user_min_samples,
                        "n_user_clusters": n_user_clusters,
                        "user_noise_ratio": user_noise,
                        "largest_user_cluster_frac": largest_user,
                        "user_silhouette": silhouette,
                        "label_ari": adjusted_rand_score(
                            user_metadata["label"], user_labels
                        ),
                        "label_nmi": normalized_mutual_info_score(
                            user_metadata["label"], user_labels
                        ),
                        "disorder_ari": adjusted_rand_score(
                            user_metadata["disorder"], user_labels
                        ),
                        "disorder_nmi": normalized_mutual_info_score(
                            user_metadata["disorder"], user_labels
                        ),
                    })
                    assignments.append(pd.DataFrame({
                        "representation": name,
                        "min_samples_message": min_samples,
                        "percentile_message": percentile,
                        "eps_message": eps_message,
                        "min_samples_user": user_min_samples,
                        "user_percentile": user_percentile,
                        "eps_user": eps_user,
                        "subject_key": user_metadata["subject_key"],
                        "user_cluster": user_labels,
                    }))
            print(
                f"{name} min_samples={min_samples} P{percentile}: "
                f"message_clusters={n_message_clusters} message_noise={message_noise:.1%}",
                flush=True,
            )

    return (
        pd.DataFrame(message_rows),
        pd.DataFrame(user_rows),
        pd.concat(assignments, ignore_index=True),
        pd.concat(message_assignments, ignore_index=True)
        if message_assignments
        else pd.DataFrame(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", nargs="+")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--vector-dir", type=Path, default=VECTOR_DIR)
    parser.add_argument(
        "--save-message-assignments",
        action="store_true",
        help="Guarda id_message y cluster para cada configuración de mensajes.",
    )
    args = parser.parse_args()
    vector_dir = args.vector_dir.resolve()
    if args.all:
        representations = sorted(
            path.name
            for path in vector_dir.glob("message__all__*")
            if (path / "X.npy").exists() and (path / "metadata.csv").exists()
        )
    elif args.representation:
        representations = args.representation
    else:
        parser.error("Indica --representation o --all")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        run_representation(name, vector_dir, args.save_message_assignments)
        for name in representations
    ]
    message = pd.concat([output[0] for output in outputs], ignore_index=True)
    user = pd.concat([output[1] for output in outputs], ignore_index=True)
    assignments = pd.concat([output[2] for output in outputs], ignore_index=True)
    message_assignments = pd.concat(
        [output[3] for output in outputs if not output[3].empty],
        ignore_index=True,
    ) if args.save_message_assignments else pd.DataFrame()
    message.to_csv(RESULT_DIR / "e2_message_hierarchical_sweep.csv", index=False)
    user.to_csv(RESULT_DIR / "e3_user_from_message_clusters.csv", index=False)
    assignments.to_csv(RESULT_DIR / "e3_user_cluster_assignments.csv", index=False)
    if args.save_message_assignments:
        message_assignments.to_csv(
            RESULT_DIR / "e2_message_cluster_assignments.csv", index=False
        )


if __name__ == "__main__":
    main()
