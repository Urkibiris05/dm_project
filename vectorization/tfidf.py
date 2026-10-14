"""Vectorización dispersa: TF-IDF seguido de SVD truncada (LSA).

El TF-IDF crudo no sirve para DBSCAN (no hay contraste de distancias y en
denso ocuparía ~12 GB), así que siempre se reduce con SVD y se normaliza en
L2, de modo que coseno y euclídea den la misma partición.
"""

import argparse

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from .units import Units

DEFAULT_NGRAMS: dict[str, tuple[int, int]] = {
    "word": (1, 1),
    "char_wb": (1, 5),
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Añade al subcomando ``tfidf`` sus opciones propias."""
    parser.add_argument("--analyzer", choices=sorted(DEFAULT_NGRAMS),
                        default="word",
                        help="palabras o n-gramas de caracteres")
    parser.add_argument("--ngram-range", nargs=2, type=int, default=None,
                        metavar=("MIN", "MAX"),
                        help="por defecto 1 1 (word) o 1 5 (char_wb)")
    parser.add_argument("--min-df", type=int, default=2,
                        help="frecuencia mínima de documento (def. 2)")
    parser.add_argument("--max-df", type=float, default=0.95,
                        help="frecuencia máxima de documento (def. 0.95)")
    parser.add_argument("--max-features", type=int, default=None,
                        help="tamaño máximo del vocabulario")
    parser.add_argument("--sublinear-tf", action="store_true",
                        help="usar 1 + log(tf)")
    parser.add_argument("--svd", type=int, default=128,
                        help="componentes de la SVD (def. 128)")


def resolve_ngram_range(args: argparse.Namespace) -> tuple[int, int]:
    """Devuelve el rango de n-gramas (por defecto el del analizador)."""
    if args.ngram_range is None:
        return DEFAULT_NGRAMS[args.analyzer]
    return args.ngram_range[0], args.ngram_range[1]


def vectorize(
    units: Units,
    fit_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, Pipeline, dict]:
    """Calcula TF-IDF + SVD + normalización L2 para cada unidad.

    El vocabulario y la SVD se ajustan solo con las unidades de
    ``fit_mask`` y se aplican a todas. Si hay menos documentos o términos
    que componentes pedidos, se reduce el número de componentes. Devuelve
    ``(matriz, pipeline_ajustado, info)``; el pipeline permite transformar
    textos nuevos más adelante.
    """
    texts = np.array(units.texts, dtype=object)
    train_texts = texts[fit_mask].tolist()
    tfidf = TfidfVectorizer(
        analyzer=args.analyzer,
        ngram_range=resolve_ngram_range(args),
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        sublinear_tf=args.sublinear_tf,
    )
    train_sparse = tfidf.fit_transform(train_texts)
    n_components = min(args.svd, train_sparse.shape[1] - 1,
                       train_sparse.shape[0] - 1)
    if n_components < 1:
        raise ValueError("Muy pocos documentos o términos para la SVD")
    svd = TruncatedSVD(n_components=n_components, random_state=args.seed)
    svd.fit(train_sparse)
    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("svd", svd),
        ("normalizer", Normalizer(norm="l2")),
    ])
    matrix = pipeline.transform(texts.tolist())
    info = {
        "vocabulary_size": len(tfidf.vocabulary_),
        "svd_components": n_components,
        "svd_explained_variance": round(
            float(svd.explained_variance_ratio_.sum()), 4),
    }
    return matrix.astype(np.float32), pipeline, info
