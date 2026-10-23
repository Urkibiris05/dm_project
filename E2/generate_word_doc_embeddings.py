"""Genera baselines Word2Vec y Doc2Vec para E2/E3."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vectorizations2" / "message__all__e5"
OUTPUT = ROOT / "vectorizations2"
VECTOR_SIZE = 100
SEED = 20261022
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def save_vectorization(
    name: str, vectors: np.ndarray, metadata: pd.DataFrame
) -> None:
    folder = OUTPUT / name
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "X.npy", vectors.astype(np.float32))
    metadata.to_csv(folder / "metadata.csv", index=False)
    (folder / "config.json").write_text(
        '{"method": "word2vec/doc2vec", "vector_size": 100}\n',
        encoding="utf-8",
    )


def main() -> None:
    metadata = pd.read_csv(SOURCE / "metadata.csv")
    documents = [tokenize(text) for text in metadata["text"]]
    if any(not document for document in documents):
        raise ValueError("Hay mensajes sin tokens")

    word_model = Word2Vec(
        sentences=documents,
        vector_size=VECTOR_SIZE,
        window=5,
        min_count=2,
        workers=4,
        epochs=10,
        seed=SEED,
    )
    word_vectors = np.vstack([
        np.mean(
            [word_model.wv[token] for token in document
             if token in word_model.wv],
            axis=0,
        )
        if any(token in word_model.wv for token in document)
        else np.zeros(VECTOR_SIZE)
        for document in documents
    ])

    tagged = [
        TaggedDocument(words=document, tags=[str(index)])
        for index, document in enumerate(documents)
    ]
    doc_model = Doc2Vec(
        documents=tagged,
        vector_size=VECTOR_SIZE,
        window=5,
        min_count=2,
        workers=4,
        epochs=20,
        dm=1,
        seed=SEED,
    )
    doc_vectors = np.vstack([
        doc_model.dv[str(index)] for index in range(len(documents))
    ])

    save_vectorization("message__all__word2vec", word_vectors, metadata)
    save_vectorization("message__all__doc2vec", doc_vectors, metadata)

    for method, vectors in (
        ("word2vec", word_vectors),
        ("doc2vec", doc_vectors),
    ):
        user_metadata = (
            metadata.drop_duplicates("subject_key")
            .set_index("subject_key")
            .loc[metadata["subject_key"].drop_duplicates()]
            .reset_index()
        )
        user_vectors = np.vstack([
            vectors[metadata["subject_key"].to_numpy() == subject].mean(axis=0)
            for subject in user_metadata["subject_key"]
        ])
        save_vectorization(
            f"user__all__{method}-mean", user_vectors, user_metadata
        )


if __name__ == "__main__":
    main()
