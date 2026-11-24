"""Export previous frozen message embeddings aligned to a common train set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_CSV = MODULE_DIR.parent / "resultados" / "messages_train.csv"
DEFAULT_EMBEDDING_ROOT = MODULE_DIR.parents[1] / "vectorizations2"
DEFAULT_OUTPUT_DIR = (
    MODULE_DIR / "reports" / "embeddings" / "transformer_common_train"
)
DISORDERS = ("Ansiedad", "Depresion", "TCA")
MODEL_ALIASES = (
    "beto-sts",
    "beto-raw",
    "minilm",
    "e5",
    "robertalex",
    "robertuito-base-uncased",
)
KEY_COLUMNS = ("trastorno", "subject_id", "id_message")
TRAIN_COLUMNS = (
    "subject_id",
    "trastorno",
    "source_partition",
    "label",
    "label_name",
    "class_name",
    "id_message",
    "date",
    "partition",
)


def _key_tuples(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    return list(
        frame.loc[:, KEY_COLUMNS].astype("string").itertuples(
            index=False,
            name=None,
        )
    )


def _read_model_cache(
    embedding_root: Path,
    alias: str,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    model_dir = embedding_root / f"message__all__{alias}"
    config_path = model_dir / "config.json"
    metadata_path = model_dir / "metadata.csv"
    embeddings_path = model_dir / "X.npy"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vectorizer = config.get("vectorizer", {})
    model_id = vectorizer.get("model")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(
            f"No se encontró el modelo Hugging Face en {config_path}"
        )

    metadata = pd.read_csv(
        metadata_path,
        dtype={
            "disorder": "string",
            "subject": "string",
            "id_message": "string",
        },
        usecols=["disorder", "subject", "id_message", "label"],
    )
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Filas de embeddings y metadata no alineadas para {alias}"
        )
    metadata = metadata.rename(
        columns={"disorder": "trastorno", "subject": "subject_id"}
    )
    if metadata[["trastorno", "subject_id", "id_message"]].isna().any().any():
        raise ValueError(f"Claves vacías en la metadata de {alias}")
    if metadata.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"Claves duplicadas en la metadata de {alias}")
    return embeddings, metadata, model_id


def export_embeddings(
    train_csv: Path,
    embedding_root: Path,
    output_dir: Path,
    model_aliases: tuple[str, ...] = MODEL_ALIASES,
) -> pd.DataFrame:
    """Write a text-free common train CSV and aligned model-specific caches."""
    train = pd.read_csv(
        train_csv,
        dtype={"subject_id": "string", "id_message": "string"},
    )
    required = set(TRAIN_COLUMNS)
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"CSV train sin columnas requeridas: {missing}")
    if train.empty or not train["partition"].astype("string").eq(
        "train"
    ).all():
        raise ValueError("El CSV de entrada debe contener solo filas de train")
    if train[list(KEY_COLUMNS)].isna().any().any():
        raise ValueError("Claves vacías en el CSV de train")
    if train.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Claves duplicadas en el CSV de train")
    unknown_disorders = sorted(
        set(train["trastorno"].dropna()) - set(DISORDERS)
    )
    if unknown_disorders:
        raise ValueError(
            f"Trastornos inesperados en train: {unknown_disorders}"
        )

    if not model_aliases or len(set(model_aliases)) != len(
        model_aliases
    ):
        raise ValueError(
            "Debe indicarse al menos un alias sin duplicados"
        )
    if output_dir.exists():
        raise FileExistsError(
            f"El directorio de salida ya existe: {output_dir}"
        )

    train_keys = _key_tuples(train)
    train_key_set = set(train_keys)
    available_keys: dict[str, set[tuple[str, str, str]]] = {}
    caches: dict[str, tuple[np.ndarray, pd.DataFrame, str]] = {}
    for alias in model_aliases:
        embeddings, metadata, model_id = _read_model_cache(
            embedding_root,
            alias,
        )
        metadata_keys = _key_tuples(metadata)
        metadata_key_set = set(metadata_keys)
        matching_keys = train_key_set & metadata_key_set
        matching_metadata = metadata.loc[
            [
                key in train_key_set
                for key in metadata_keys
            ],
            ["trastorno", "subject_id", "id_message", "label"],
        ]
        train_labels = train.set_index(list(KEY_COLUMNS))["label"]
        embedding_labels = matching_metadata.set_index(
            list(KEY_COLUMNS)
        )["label"]
        for key in matching_keys:
            if int(train_labels.loc[key]) != int(embedding_labels.loc[key]):
                raise ValueError(
                    f"Etiqueta de sujeto discrepante en {alias} para {key}"
                )
        available_keys[alias] = matching_keys
        caches[alias] = (embeddings, metadata, model_id)

    common_keys = train_key_set.intersection(*available_keys.values())
    common_mask = [key in common_keys for key in train_keys]
    common_train = train.loc[common_mask, list(TRAIN_COLUMNS)].reset_index(
        drop=True
    )
    if common_train.empty:
        raise ValueError("No hay mensajes comunes entre los modelos y train")
    common_key_order = _key_tuples(common_train)

    output_dir.mkdir(parents=True)
    common_train.to_csv(output_dir / "common_train.csv", index=False)
    reports: list[dict[str, object]] = []
    for alias in model_aliases:
        embeddings, metadata, model_id = caches[alias]
        metadata_keys = _key_tuples(metadata)
        positions = {key: index for index, key in enumerate(metadata_keys)}
        ordered_positions = np.fromiter(
            (positions[key] for key in common_key_order),
            dtype=np.int64,
            count=len(common_key_order),
        )
        model_output = output_dir / alias
        model_output.mkdir()
        for disorder in DISORDERS:
            disorder_mask = common_train["trastorno"].eq(disorder).to_numpy()
            disorder_train = common_train.loc[disorder_mask]
            disorder_positions = ordered_positions[disorder_mask]
            cache_path = model_output / f"{disorder}.npz"
            np.savez_compressed(
                cache_path,
                embeddings=np.asarray(
                    embeddings[disorder_positions],
                    dtype=np.float32,
                ),
                subject_id=disorder_train["subject_id"].to_numpy(dtype=str),
                id_message=disorder_train["id_message"].to_numpy(dtype=str),
                disorder=np.array(disorder),
                model_id=np.array(model_id),
            )
        reports.append(
            {
                "alias": alias,
                "model_id": model_id,
                "dimension": int(embeddings.shape[1]),
                "train_rows": len(train),
                "available_train_rows": len(available_keys[alias]),
                "missing_train_rows": len(
                    train_key_set - available_keys[alias]
                ),
                "common_train_rows": len(common_train),
                "common_rows_by_disorder": json.dumps(
                    common_train["trastorno"].value_counts().to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

    report = pd.DataFrame(reports)
    report.to_csv(output_dir / "alignment_report.csv", index=False)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=DEFAULT_EMBEDDING_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ALIASES,
        default=MODEL_ALIASES,
    )
    args = parser.parse_args()
    report = export_embeddings(
        args.train_csv,
        args.embedding_root,
        args.output_dir,
        tuple(args.models),
    )
    print(report.to_string(index=False))
    print(f"\nAlineamiento y cachés guardados en: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
