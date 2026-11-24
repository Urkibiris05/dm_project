"""Build a train-only TF-IDF/LSA embedding cache for a quick k-NN trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_CSV = MODULE_DIR.parent / "resultados" / "messages_train.csv"
DEFAULT_OUTPUT = MODULE_DIR / "reports" / "embeddings" / "train_tfidf_svd.npz"


def build_tfidf_cache(
    train_csv: Path,
    output_path: Path,
    *,
    disorder: str,
    components: int,
    random_state: int,
    max_features: int,
    min_df: int,
) -> dict[str, Any]:
    """Fit word TF-IDF and truncated SVD on train rows only."""
    messages = pd.read_csv(
        train_csv,
        dtype={"subject_id": "string", "id_message": "string"},
    )
    required = {
        "subject_id",
        "id_message",
        "message",
        "partition",
        "trastorno",
    }
    missing = sorted(required - set(messages.columns))
    if missing:
        raise ValueError(f"CSV sin columnas requeridas: {missing}")
    if messages.empty:
        raise ValueError("El CSV no contiene mensajes")
    if not messages["partition"].astype("string").eq("train").all():
        raise ValueError("Solo se permite ajustar la representación en train")
    messages = messages.loc[messages["trastorno"].eq(disorder)].reset_index(
        drop=True
    )
    if messages.empty:
        raise ValueError(f"No hay mensajes train para el trastorno {disorder}")
    if messages[["subject_id", "id_message"]].isna().any().any():
        raise ValueError("El CSV contiene IDs vacíos")
    if messages.duplicated(["subject_id", "id_message"]).any():
        raise ValueError(
            "El CSV contiene IDs de mensaje duplicados por sujeto"
        )
    if components < 1:
        raise ValueError("components debe ser >= 1")
    if max_features < 1 or min_df < 1:
        raise ValueError("max_features y min_df deben ser >= 1")
    if output_path.exists():
        raise FileExistsError(
            f"La caché ya existe y no se sobrescribirá: {output_path}"
        )

    texts = messages["message"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    sparse_matrix = vectorizer.fit_transform(texts)
    if sparse_matrix.shape[1] < 2:
        raise ValueError(
            "TF-IDF generó menos de dos características; no se puede aplicar "
            "SVD"
        )
    actual_components = min(
        components,
        sparse_matrix.shape[1] - 1,
        sparse_matrix.shape[0] - 1,
    )
    if actual_components < 1:
        raise ValueError("No hay suficientes filas/características para SVD")

    reducer = TruncatedSVD(
        n_components=actual_components,
        n_iter=7,
        random_state=random_state,
    )
    embeddings = reducer.fit_transform(sparse_matrix).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        subject_id=messages["subject_id"].to_numpy(dtype=str),
        id_message=messages["id_message"].to_numpy(dtype=str),
        disorder=np.array(disorder),
        model_id=np.array(
            "train-only-word-tfidf-1-2gram+truncated-svd"
            f"-{actual_components}d-{disorder}-seed{random_state}"
        ),
    )
    metadata: dict[str, Any] = {
        "source_csv": str(train_csv),
        "output_cache": str(output_path),
        "model_id": (
            "train-only-word-tfidf-1-2gram+truncated-svd"
            f"-{actual_components}d-{disorder}-seed{random_state}"
        ),
        "disorder": disorder,
        "n_train_messages": int(sparse_matrix.shape[0]),
        "tfidf_features": int(sparse_matrix.shape[1]),
        "embedding_dimensions": int(embeddings.shape[1]),
        "explained_variance_ratio_sum": float(
            reducer.explained_variance_ratio_.sum()
        ),
        "random_state": random_state,
        "max_features": max_features,
        "min_df": min_df,
        "fitted_on": "train only",
        "interpretation": (
            "Exploratory lexical LSA representation, not a Sentence "
            "Transformer or contextual embedding."
        ),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--disorder",
        choices=("Ansiedad", "Depresion", "TCA"),
        required=True,
    )
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=2023)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--min-df", type=int, default=2)
    args = parser.parse_args()
    if not re.fullmatch(r".+\.npz", args.output.name):
        parser.error("--output debe tener extensión .npz")

    metadata = build_tfidf_cache(
        args.train_csv,
        args.output,
        disorder=args.disorder,
        components=args.components,
        random_state=args.random_state,
        max_features=args.max_features,
        min_df=args.min_df,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
