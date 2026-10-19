"""E1.4: evaluación externa e interpretación de las configuraciones elegidas.

Las configuraciones ya están fijadas por E1.3 (sin etiquetas). Aquí, y solo
aquí, se comparan con las etiquetas Gold, que son por usuario: a nivel de
mensaje cada mensaje hereda la etiqueta de su usuario, así que los valores
de ARI/AMI tienen un techo bajo (el acuerdo entre anotadores es moderado).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from evaluation import NOISE
from evaluation.metrics import external_metrics

from . import config, datasets
from .exp_grid import fit_dbscan

N_TERMS: int = 10
N_NEAREST: int = 3
MAX_CLUSTERS_DESCRIBED: int = 10
TEXT_CHARS: int = 200


def ctfidf_terms(texts_by_cluster: dict[int, list[str]]) -> dict[int, str]:
    """Términos más característicos de cada cluster (c-TF-IDF).

    Se junta el texto de cada cluster en un documento. Peso de un término t
    en el cluster c: ``tf(t, c) * log(1 + A / f(t))``, con A el número medio
    de palabras por cluster y f(t) la frecuencia total del término
    (Grootendorst, 2022).
    """
    ids = list(texts_by_cluster)
    counter = CountVectorizer()
    counts = counter.fit_transform(
        [" ".join(texts_by_cluster[i]) for i in ids]).toarray()
    vocabulary = np.array(counter.get_feature_names_out())
    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    average = counts.sum() / len(ids)
    weights = tf * np.log(1 + average / np.maximum(counts.sum(axis=0), 1))
    return {
        cluster: ", ".join(vocabulary[np.argsort(-weights[row])[:N_TERMS]])
        for row, cluster in enumerate(ids)
    }


def describe_clusters(
    X: np.ndarray, labels: np.ndarray, metadata: pd.DataFrame, info: dict
) -> list[dict]:
    """Tamaño, composición, términos y textos centrales de cada cluster."""
    sizes = pd.Series(labels[labels != NOISE]).value_counts()
    biggest = list(sizes.index[:MAX_CLUSTERS_DESCRIBED])
    texts = metadata["text"].fillna("").to_numpy()
    terms = ctfidf_terms({c: list(texts[labels == c]) for c in biggest})
    rows = []
    for cluster in biggest:
        members = np.flatnonzero(labels == cluster)
        centroid = X[members].mean(axis=0)
        # X está normalizada: el más cercano al centroide es el de mayor
        # producto escalar.
        order = members[np.argsort(-(X[members] @ centroid))]
        rows.append({
            **info, "cluster": int(cluster), "size": len(members),
            "frac_label1": float(metadata["label"].to_numpy()[members]
                                 .astype(float).mean()),
            "terms_ctfidf": terms[cluster],
            "central_texts": " || ".join(
                t[:TEXT_CHARS] for t in texts[order[:N_NEAREST]]),
        })
    return rows


def main(rapido: bool = False) -> None:
    chosen = pd.read_csv(config.RESULTS_DIR / "e13_seleccion.csv")
    external_rows: list[dict] = []
    cluster_rows: list[dict] = []
    for _, row in chosen.iterrows():
        space = {key: row[key] for key in config.SPACE_COLUMNS}
        space["svd"] = int(space["svd"])
        X, metadata = datasets.load_for(space, rapido)
        labels, _ = fit_dbscan(X, float(row["eps"]), int(row["min_samples"]))
        info = {**space, "eps": row["eps"],
                "min_samples": int(row["min_samples"]),
                "status": row["status"]}
        known = metadata["label"].notna().to_numpy()
        external_rows.append({
            **info, "target_type": "label",
            **external_metrics(labels[known],
                               metadata["label"].to_numpy()[known])})
        if space["disorder"] == "Depresion" and "target" in metadata:
            known = metadata["target"].notna().to_numpy()
            external_rows.append({
                **info, "target_type": "target",
                **external_metrics(labels[known],
                                   metadata["target"].to_numpy()[known])})
        if (labels != NOISE).any():
            cluster_rows.extend(
                describe_clusters(X, labels, metadata, info))
    pd.DataFrame(external_rows).to_csv(
        config.RESULTS_DIR / "e14_externas.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(
        config.RESULTS_DIR / "e14_clusters.csv", index=False)
