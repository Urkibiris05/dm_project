"""Generate complete BETO-STS message embeddings for selected shared splits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = MODULE_DIR.parent / "resultados"
DEFAULT_OUTPUT_DIR = (
    MODULE_DIR / "reports" / "embeddings" / "beto_sts_full_baseline"
)
MODEL_ID = "hiiamsid/sentence_similarity_spanish_es"
MODEL_REVISION = "66ab46adac3910bb6ea6085b962a25e49513b981"
PARTITIONS = ("train", "dev", "test")
DEFAULT_TORCH_THREADS = min(14, os.cpu_count() or 1)
REQUIRED_COLUMNS = {
    "subject_id",
    "trastorno",
    "label",
    "id_message",
    "message",
    "date",
    "partition",
}


def _validate_messages(frame: pd.DataFrame, partition: str) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{partition}: faltan columnas {missing}")
    if frame.empty:
        raise ValueError(f"{partition}: no hay mensajes")
    if not frame["partition"].astype("string").eq(partition).all():
        raise ValueError(
            f"{partition}: el CSV contiene filas de otra partición"
        )
    if frame[["subject_id", "id_message"]].isna().any().any():
        raise ValueError(f"{partition}: IDs de sujeto/mensaje vacíos")
    if frame.duplicated(["trastorno", "subject_id", "id_message"]).any():
        raise ValueError(f"{partition}: IDs de mensaje duplicados")
    if frame["label"].isna().any() or not frame["label"].isin([0, 1]).all():
        raise ValueError(f"{partition}: etiquetas nulas o no binarias")
    if frame["message"].isna().any():
        raise ValueError(f"{partition}: textos de mensaje nulos")


def _load_encoder(device: str, torch_threads: int) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "BETO-STS requiere sentence-transformers y torch; usa el "
            "entorno del proyecto e instala vectorization/requirements.txt "
            "si faltan."
        ) from exc
    import torch

    torch.set_num_threads(torch_threads)
    return SentenceTransformer(
        MODEL_ID,
        device=device,
        revision=MODEL_REVISION,
        local_files_only=True,
    )


def encode_partitions(
    *,
    data_dir: Path,
    output_dir: Path,
    partitions: tuple[str, ...] = ("train", "dev"),
    batch_size: int = 32,
    device: str = "cpu",
    torch_threads: int = DEFAULT_TORCH_THREADS,
    encoder: Any | None = None,
) -> dict[str, Any]:
    """Encode all messages in the requested splits without persisting text."""
    if not partitions or len(set(partitions)) != len(partitions):
        raise ValueError("Indica particiones únicas")
    invalid = sorted(set(partitions) - set(PARTITIONS))
    if invalid:
        raise ValueError(f"Particiones no permitidas: {invalid}")
    if batch_size < 1:
        raise ValueError("batch_size debe ser >= 1")
    if torch_threads < 1:
        raise ValueError("torch_threads debe ser >= 1")
    if output_dir.exists():
        raise FileExistsError(
            f"El directorio ya existe y no se sobrescribirá: {output_dir}"
        )

    frames: dict[str, pd.DataFrame] = {}
    for partition in partitions:
        path = data_dir / f"messages_{partition}.csv"
        frame = pd.read_csv(
            path,
            dtype={"subject_id": "string", "id_message": "string"},
        )
        _validate_messages(frame, partition)
        frames[partition] = frame

    model = (
        encoder
        if encoder is not None
        else _load_encoder(device, torch_threads)
    )
    if encoder is not None:
        model_id = getattr(encoder, "model_id", MODEL_ID)
    else:
        model_id = MODEL_ID

    output_dir.mkdir(parents=True)
    split_summary: dict[str, dict[str, Any]] = {}
    for partition, frame in frames.items():
        texts = frame["message"].astype(str).tolist()
        vectors = np.asarray(
            model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )
        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(frame)
            or vectors.shape[1] == 0
        ):
            raise ValueError(
                f"{partition}: el encoder devolvió una matriz "
                f"desalineada con forma {vectors.shape}"
            )
        if not np.isfinite(vectors).all():
            raise ValueError(
                f"{partition}: el encoder produjo valores no finitos"
            )

        output_path = output_dir / f"{partition}.npz"
        np.savez_compressed(
            output_path,
            embeddings=vectors,
            subject_id=frame["subject_id"].to_numpy(dtype=str),
            id_message=frame["id_message"].to_numpy(dtype=str),
            disorder=frame["trastorno"].to_numpy(dtype=str),
            label=frame["label"].to_numpy(dtype=np.int64),
            date=frame["date"].astype(str).to_numpy(dtype=str),
            partition=np.array(partition),
            model_id=np.array(model_id),
        )
        split_summary[partition] = {
            "messages": int(len(frame)),
            "subjects": int(frame["subject_id"].nunique()),
            "dimensions": int(vectors.shape[1]),
            "output": str(output_path),
        }

    manifest = {
        "model_id": model_id,
        "model_revision": MODEL_REVISION,
        "representation": "SentenceTransformer sentence_embedding",
        "pooling": "model-configured mean pooling over non-padding tokens",
        "max_sequence_length": getattr(model, "max_seq_length", None),
        "fine_tuned_on_task_labels": False,
        "normalized_embeddings": False,
        "device": device,
        "torch_threads": torch_threads,
        "batch_size": batch_size,
        "min_tokens_filter": None,
        "message_texts_written_to_outputs": False,
        "splits": split_summary,
        "note": (
            "Frozen encoder inference. Test is not included unless explicitly "
            "requested after the final protocol is frozen."
        ),
    }
    manifest_path = output_dir / "embedding_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=PARTITIONS,
        default=("train", "dev"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=DEFAULT_TORCH_THREADS,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    manifest = encode_partitions(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        partitions=tuple(args.partitions),
        batch_size=args.batch_size,
        device=args.device,
        torch_threads=args.torch_threads,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
