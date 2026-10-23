"""E8: asigna mensajes Test a clusters DBSCAN aprendidos en Train."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from dbscan_implementation import DBSCAN


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "resultados"


def external_scores(metadata: pd.DataFrame, labels: np.ndarray) -> dict:
    return {
        "label_ari": adjusted_rand_score(
            metadata["label"].astype("category").cat.codes, labels
        ),
        "label_nmi": normalized_mutual_info_score(
            metadata["label"].astype("category").cat.codes, labels
        ),
    }


def contingency_table(
    test_metadata: pd.DataFrame, assigned: np.ndarray
) -> pd.DataFrame:
    """Cuenta etiquetas reales por cluster asignado, incluido el ruido."""
    return pd.crosstab(
        pd.Series(test_metadata["label"].to_numpy(), name="label"),
        pd.Series(assigned, name="cluster_assigned"),
        dropna=False,
    )


def optimal_cluster_mapping(contingency: pd.DataFrame) -> pd.DataFrame:
    """Obtiene la correspondencia uno-a-uno que maximiza coincidencias."""
    clusters = [
        int(cluster) for cluster in contingency.columns if int(cluster) >= 0
    ]
    labels = list(contingency.index)
    columns = [
        "cluster",
        "label",
        "support",
        "cluster_size",
        "label_size",
        "precision",
        "recall",
    ]
    if not clusters or not labels:
        return pd.DataFrame(columns=columns)

    counts = contingency.reindex(columns=clusters, fill_value=0).to_numpy()
    label_rows, cluster_columns = linear_sum_assignment(counts, maximize=True)
    rows = []
    for label_row, cluster_column in zip(label_rows, cluster_columns):
        cluster = clusters[cluster_column]
        support = int(counts[label_row, cluster_column])
        cluster_size = int(counts[:, cluster_column].sum())
        label_size = int(counts[label_row, :].sum())
        rows.append({
            "cluster": cluster,
            "label": labels[label_row],
            "support": support,
            "cluster_size": cluster_size,
            "label_size": label_size,
            "precision": support / cluster_size if cluster_size else np.nan,
            "recall": support / label_size if label_size else np.nan,
        })
    return pd.DataFrame(rows, columns=columns)


def instance_assignments(
    test_metadata: pd.DataFrame,
    assigned: np.ndarray,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Añade al metadata de Test el cluster y la correspondencia esperada."""
    result = test_metadata.reset_index(names="original_index").copy()
    result["cluster_assigned"] = assigned
    result["is_noise"] = assigned == -1
    label_to_cluster = mapping.set_index("label")["cluster"]
    result["expected_cluster"] = result["label"].map(label_to_cluster)
    result["expected_cluster"] = result["expected_cluster"].astype("Int64")
    result["matches_expected_cluster"] = (
        result["expected_cluster"].notna()
        & (result["cluster_assigned"] == result["expected_cluster"])
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", default="message__all__e5")
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--percentile", type=int, default=40)
    parser.add_argument("--output-prefix", default="e8")
    args = parser.parse_args()
    folder = ROOT / "vectorizations2" / args.representation
    X = np.load(folder / "X.npy", mmap_mode="r")
    metadata = pd.read_csv(folder / "metadata.csv")
    train = metadata["split"].astype(str).str.lower().eq("train").to_numpy()
    test = metadata["split"].astype(str).str.lower().eq("test").to_numpy()
    train_idx = np.flatnonzero(train)
    test_idx = np.flatnonzero(test)
    train_X, test_X = X[train_idx], X[test_idx]
    distances, _ = NearestNeighbors(
        n_neighbors=args.min_samples,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    ).fit(train_X).kneighbors(train_X)
    eps = float(np.percentile(distances[:, -1], args.percentile))
    model = DBSCAN(
        eps=eps,
        min_samples=args.min_samples,
        metric="cosine",
        algorithm="optimized_blocked",
        deduplicate_queue=True,
    ).fit(train_X)
    assigned = model.assign(test_X)
    test_metadata = metadata.iloc[test_idx].copy()
    contingency = contingency_table(test_metadata, assigned)
    mapping = optimal_cluster_mapping(contingency)
    assignments = instance_assignments(test_metadata, assigned, mapping)
    row = {
        "experiment": args.output_prefix,
        "representation": args.representation,
        "metric": "cosine",
        "min_samples": args.min_samples,
        "percentile": args.percentile,
        "eps": eps,
        "n_train": len(train_X),
        "n_test": len(test_X),
        "train_clusters": len(set(model.labels_)) - (
            1 if -1 in model.labels_ else 0
        ),
        "train_noise_ratio": float(np.mean(model.labels_ == -1)),
        "test_noise_ratio": float(np.mean(assigned == -1)),
        "test_assigned_ratio": float(np.mean(assigned >= 0)),
        **external_scores(metadata.iloc[test_idx], assigned),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / f"{args.output_prefix}_message_assignment.csv"
    pd.DataFrame([row]).to_csv(output, index=False)
    assignments.to_csv(
        RESULTS / f"{args.output_prefix}_message_assignments.csv",
        index=False,
    )
    contingency.to_csv(RESULTS / f"{args.output_prefix}_message_contingency.csv")
    mapping.to_csv(
        RESULTS / f"{args.output_prefix}_cluster_label_mapping.csv",
        index=False,
    )
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
