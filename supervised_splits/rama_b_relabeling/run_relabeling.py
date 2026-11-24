"""Run one k-NN relabeling configuration on cached train embeddings."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from relabeling import RelabelResult, relabel_messages


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_CSV = MODULE_DIR.parent / "resultados" / "messages_train.csv"
OUTPUT_ROOT = MODULE_DIR / "reports" / "relabeling"
DISORDERS = ("Ansiedad", "Depresion", "TCA")
OUTPUT_COLUMNS = [
    "subject_id",
    "id_message",
    "initial_label",
    "weak_label",
    "was_relabelled",
]


def load_aligned_train_data(
    train_csv: Path,
    embedding_cache: Path,
    disorder: str,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Load a train CSV and NPZ cache, rejecting row-order mismatches."""
    messages = pd.read_csv(
        train_csv,
        dtype={"subject_id": "string", "id_message": "string"},
    )
    required = {
        "subject_id",
        "id_message",
        "label",
        "partition",
        "trastorno",
    }
    missing = sorted(required - set(messages.columns))
    if missing:
        raise ValueError(f"CSV sin columnas requeridas: {missing}")
    if messages.empty:
        raise ValueError("El CSV de train no contiene mensajes")
    if not messages["partition"].astype("string").eq("train").all():
        raise ValueError("El CSV puede contener únicamente filas de train")
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

    try:
        with np.load(embedding_cache, allow_pickle=False) as cache:
            cache_keys = set(cache.files)
            required_cache = {
                "embeddings",
                "subject_id",
                "id_message",
                "disorder",
            }
            missing_cache = sorted(required_cache - cache_keys)
            if missing_cache:
                raise ValueError(
                    f"NPZ sin arrays requeridos: {missing_cache}"
                )
            vectors = np.asarray(cache["embeddings"], dtype=np.float64)
            cached_subjects = np.asarray(cache["subject_id"])
            cached_message_ids = np.asarray(cache["id_message"])
            cached_disorder = str(np.asarray(cache["disorder"]).item())
            model_id = (
                str(np.asarray(cache["model_id"]).item())
                if "model_id" in cache_keys
                else "unspecified"
            )
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(
            f"No se pudo leer la caché de embeddings {embedding_cache}: {exc}"
        ) from exc

    expected_subjects = messages["subject_id"].astype(str).to_numpy()
    expected_message_ids = messages["id_message"].astype(str).to_numpy()
    if vectors.ndim != 2 or vectors.shape[0] != len(messages):
        raise ValueError(
            "La caché debe tener una matriz 2D con una fila por mensaje de "
            "train"
        )
    if not np.array_equal(cached_subjects.astype(str), expected_subjects):
        raise ValueError(
            "El orden de subject_id de la caché no coincide con el CSV"
        )
    aligned_message_ids = np.array_equal(
        cached_message_ids.astype(str),
        expected_message_ids,
    )
    if not aligned_message_ids:
        raise ValueError(
            "El orden de id_message de la caché no coincide con el CSV"
        )
    if cached_disorder != disorder:
        raise ValueError(
            f"La caché es de {cached_disorder}, no de {disorder}"
        )
    return messages, vectors, model_id


def write_run(
    messages: pd.DataFrame,
    result: RelabelResult,
    *,
    output_dir: Path,
    disorder: str,
    model_id: str,
    neighbors: int,
    min_control_neighbors: int,
    mode: str,
    exclude_same_subject: bool,
    max_neighbors_per_subject: int | None,
    max_rounds: int | None,
) -> None:
    """Write privacy-conscious labels, round history, and run metadata."""
    output_dir.mkdir(parents=True, exist_ok=False)
    output_frame = messages[["subject_id", "id_message"]].copy()
    output_frame["initial_label"] = messages["label"].astype("int64")
    output_frame["weak_label"] = result.labels
    output_frame["was_relabelled"] = (
        output_frame["initial_label"] != output_frame["weak_label"]
    )
    output_frame[OUTPUT_COLUMNS].to_csv(
        output_dir / "weak_message_labels.csv",
        index=False,
    )
    pd.DataFrame(
        [round_summary.__dict__ for round_summary in result.history]
    ).to_csv(output_dir / "round_history.csv", index=False)
    manifest: dict[str, Any] = {
        "model_id": model_id,
        "disorder": disorder,
        "neighbors_J": neighbors,
        "min_control_neighbors_K": min_control_neighbors,
        "mode": mode,
        "exclude_same_subject": exclude_same_subject,
        "max_neighbors_per_subject": max_neighbors_per_subject,
        "max_rounds": max_rounds,
        "n_messages": int(len(messages)),
        "initial_suffer": int((output_frame["initial_label"] == 1).sum()),
        "initial_control": int((output_frame["initial_label"] == 0).sum()),
        "relabelled_suffer_to_control": int(
            output_frame["was_relabelled"].sum()
        ),
        "final_suffer": int((output_frame["weak_label"] == 1).sum()),
        "final_control": int((output_frame["weak_label"] == 0).sum()),
        "note": (
            "Weak message labels only; not human-validated gold labels."
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument(
        "--disorder",
        choices=DISORDERS,
        required=True,
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        required=True,
        help=(
            "NPZ with embeddings, subject_id and id_message arrays in exactly "
            "the same row order as train-csv; optional model_id scalar."
        ),
    )
    parser.add_argument("--neighbors", type=int, required=True, help="J")
    parser.add_argument(
        "--min-control-neighbors",
        type=int,
        required=True,
        help=(
            "K: minimum negative neighbors needed to relabel suffer as "
            "control"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("single_pass", "iterative"),
        default="single_pass",
    )
    parser.add_argument("--exclude-same-subject", action="store_true")
    parser.add_argument("--max-neighbors-per-subject", type=int)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--run-name",
        default="latest",
        help=(
            "Output folder name, written under this branch's reports "
            "directory."
        ),
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        parser.error("--run-name solo puede contener letras, números, . _ y -")
    return args


def main() -> int:
    args = _parse_args()
    messages, vectors, model_id = load_aligned_train_data(
        args.train_csv,
        args.embedding_cache,
        args.disorder,
    )
    result = relabel_messages(
        vectors,
        messages["label"].to_numpy(dtype=np.int64),
        messages["subject_id"].astype(str).to_numpy(),
        neighbors=args.neighbors,
        min_control_neighbors=args.min_control_neighbors,
        mode=args.mode,
        exclude_same_subject=args.exclude_same_subject,
        max_neighbors_per_subject=args.max_neighbors_per_subject,
        max_rounds=args.max_rounds,
        block_size=args.block_size,
    )
    output_dir = OUTPUT_ROOT / args.run_name
    write_run(
        messages,
        result,
        output_dir=output_dir,
        disorder=args.disorder,
        model_id=model_id,
        neighbors=args.neighbors,
        min_control_neighbors=args.min_control_neighbors,
        mode=args.mode,
        exclude_same_subject=args.exclude_same_subject,
        max_neighbors_per_subject=args.max_neighbors_per_subject,
        max_rounds=args.max_rounds,
    )
    changed = result.history[-1].changed_total if result.history else 0
    print(f"Relabeled {changed} message(s) from suffer to control.")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
