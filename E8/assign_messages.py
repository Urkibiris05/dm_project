"""Guarda un DBSCAN BETO-STS y asigna mensajes nuevos desde un archivo TXT."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from dbscan_implementation import DBSCAN
from dbscan_implementation.distances import get_distance
from vectorization.cache import VectorCache
from vectorization.combine import l2_normalize
from vectorization.preprocessing import (
    build_emoji_index,
    load_emoji_names,
    preprocess,
)
from vectorization.transformer import (
    MODEL_ALIASES,
    embed_texts,
    make_encoder,
    resolve_model,
)
from vectorization.transformer import cache_path as transformer_cache_path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VECTOR_DIR = ROOT / "vectorizations2"
DEFAULT_MODEL_DIR = ROOT / "E8" / "modelos"
MODEL_FORMAT = 1
MODEL_ALIAS = "beto-sts"
MIN_TOKENS = 4
METRIC = "cosine"
ALGORITHM = "optimized_blocked"


def default_model_path(
    representation: str,
    min_samples: int,
    percentile: int,
) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", representation)
    return DEFAULT_MODEL_DIR / (
        f"{slug}_dbscan_m{min_samples}_p{percentile}.joblib"
    )


def prepare_assigner(
    representation: str,
    vector_dir: Path,
    model_path: Path,
    min_samples: int,
    percentile: int,
) -> dict[str, Any]:
    """Ajusta DBSCAN solo con Train y persiste el modelo para asignación."""
    if min_samples < 1:
        raise ValueError("--min-samples debe ser al menos 1")
    if not 0 <= percentile <= 100:
        raise ValueError("--percentile debe estar entre 0 y 100")

    folder = vector_dir / representation
    config_path = folder / "config.json"
    if (
        not (folder / "X.npy").is_file()
        or not (folder / "metadata.csv").is_file()
    ):
        raise FileNotFoundError(
            f"Faltan X.npy o metadata.csv en la representación {folder}"
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"No existe la configuración {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    vector_args = config.get("args", {})
    if (
        config.get("command") != "transformer"
        or vector_args.get("model") != MODEL_ALIAS
        or vector_args.get("unit") != "message"
        or vector_args.get("no_normalize")
        or vector_args.get("affect", 0) != 0
        or vector_args.get("time", 0) != 0
    ):
        raise ValueError(
            "La demo requiere una representación de mensajes transformer "
            "BETO-STS normalizada, sin bloques adicionales."
        )

    X = np.load(folder / "X.npy", mmap_mode="r")
    metadata = pd.read_csv(folder / "metadata.csv")
    if len(X) != len(metadata):
        raise ValueError(
            "X.npy y metadata.csv no tienen el mismo número de filas"
        )
    train_mask = (
        metadata["split"].astype(str).str.lower().eq("train").to_numpy()
    )
    train_X = X[train_mask]
    if len(train_X) < min_samples:
        raise ValueError(
            f"Train tiene {len(train_X)} filas; min_samples={min_samples} "
            "no es válido."
        )

    neighbor_distances, _ = NearestNeighbors(
        n_neighbors=min_samples,
        metric=METRIC,
        algorithm="brute",
        n_jobs=-1,
    ).fit(train_X).kneighbors(train_X)
    eps = float(np.percentile(neighbor_distances[:, -1], percentile))
    if eps <= 0:
        eps = float(np.nextafter(0.0, 1.0))

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=METRIC,
        algorithm=ALGORITHM,
        deduplicate_queue=True,
    ).fit(train_X)
    n_clusters = len(set(model.labels_)) - int(-1 in model.labels_)
    artifact = {
        "format_version": MODEL_FORMAT,
        "model": model,
        "representation": representation,
        "embedding_alias": MODEL_ALIAS,
        "embedding_model": MODEL_ALIASES[MODEL_ALIAS],
        "metric": METRIC,
        "min_samples": min_samples,
        "percentile": percentile,
        "eps": eps,
        "n_features": int(train_X.shape[1]),
        "n_train": int(len(train_X)),
        "n_clusters": int(n_clusters),
        "train_noise_ratio": float(np.mean(model.labels_ == -1)),
        "cluster_sizes": {
            int(cluster): int(np.count_nonzero(model.labels_ == cluster))
            for cluster in np.unique(model.labels_)
            if cluster >= 0
        },
        "min_tokens": MIN_TOKENS,
        "normalization": "l2",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    return artifact


def read_messages(input_path: Path) -> list[tuple[int, str]]:
    """Lee un mensaje por línea no vacía, conservando el número de línea."""
    lines = input_path.read_text(encoding="utf-8-sig").splitlines()
    return [
        (line_number, line.strip())
        for line_number, line in enumerate(lines, start=1)
        if line.strip()
    ]


def vectorize_messages(
    messages: list[tuple[int, str]],
    cache_dir: Path,
    batch_size: int,
    device: str,
) -> tuple[list[tuple[int, str]], np.ndarray]:
    """Aplica el filtro de entrenamiento y vectoriza con BETO-STS."""
    frame = pd.DataFrame({
        "input_line": [line_number for line_number, _ in messages],
        "subject_key": [f"demo/{line_number}" for line_number, _ in messages],
        "message": [text for _, text in messages],
    })
    emoji_index = build_emoji_index(load_emoji_names())
    frame, _ = preprocess(
        frame,
        min_tokens=MIN_TOKENS,
        count_emoji_tokens=False,
        keep_duplicates=True,
        emojis="keep",
        emoji_index=emoji_index,
    )
    accepted = list(zip(
        frame["input_line"].astype(int).tolist(),
        frame["message"].astype(str).tolist(),
    ))
    if not accepted:
        return [], np.empty((0, 0), dtype=np.float32)

    hf_name, prefix = resolve_model(MODEL_ALIAS)
    cache = VectorCache(
        transformer_cache_path(cache_dir, hf_name, truncate_dim=None)
    )
    encoder = make_encoder(
        hf_name,
        batch_size=batch_size,
        truncate_dim=None,
        device=device,
    )
    embeddings = embed_texts(
        [text for _, text in accepted],
        prefix,
        cache,
        encoder,
    )
    return accepted, l2_normalize(embeddings).astype(np.float32)


def assign_messages(
    input_path: Path,
    model_path: Path,
    output_path: Path,
    cache_dir: Path,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    """Vectoriza TXT y asigna sus líneas válidas al DBSCAN persistido."""
    artifact = joblib.load(model_path)
    if (
        not isinstance(artifact, dict)
        or artifact.get("format_version") != MODEL_FORMAT
    ):
        raise ValueError(f"Formato de modelo no compatible: {model_path}")
    if artifact.get("embedding_alias") != MODEL_ALIAS:
        raise ValueError("El artefacto no usa el encoder BETO-STS esperado.")
    model = artifact.get("model")
    if not isinstance(model, DBSCAN):
        raise ValueError("El artefacto no contiene un modelo DBSCAN válido.")

    messages = read_messages(input_path)
    if not messages:
        raise ValueError(f"No hay mensajes no vacíos en {input_path}")
    accepted, X = vectorize_messages(messages, cache_dir, batch_size, device)
    if not accepted:
        raise ValueError(
            f"Ningún mensaje supera el filtro de {MIN_TOKENS} tokens."
        )
    if X.shape[1] != artifact["n_features"]:
        raise ValueError(
            f"BETO-STS generó {X.shape[1]} dimensiones; el modelo espera "
            f"{artifact['n_features']}."
        )

    labels = model.assign(X)
    cluster_sizes = artifact.get("cluster_sizes")
    if cluster_sizes is None:
        cluster_sizes = {
            int(cluster): int(np.count_nonzero(model.labels_ == cluster))
            for cluster in np.unique(model.labels_)
            if cluster >= 0
        }

    if len(model.components_):
        core_distances = get_distance(artifact["metric"]).pairwise(
            X, model.components_
        )
        nearest_core_indices = np.argmin(core_distances, axis=1)
        nearest_core_distances = core_distances[
            np.arange(len(X)), nearest_core_indices
        ]
        nearest_core_clusters = model.core_sample_labels_[
            nearest_core_indices
        ]
        distance_over_eps = nearest_core_distances / float(artifact["eps"])
    else:
        nearest_core_distances = np.full(len(X), np.nan)
        nearest_core_clusters = np.full(len(X), -1, dtype=int)
        distance_over_eps = np.full(len(X), np.nan)

    results = pd.DataFrame({
        "input_line": [line_number for line_number, _ in accepted],
        "text": [text for _, text in accepted],
        "cluster_assigned": labels,
        "is_noise": labels == -1,
        "nearest_core_cluster": nearest_core_clusters,
        "nearest_core_distance": nearest_core_distances,
        "distance_over_eps": distance_over_eps,
        "assigned_cluster_train_size": [
            int(cluster_sizes.get(int(cluster), 0))
            for cluster in labels
        ],
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara un modelo BETO-STS una vez o asigna mensajes de un TXT "
            "a un modelo DBSCAN ya guardado."
        )
    )
    commands = parser.add_subparsers(dest="action", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="Ajusta DBSCAN con Train y guarda el artefacto.",
    )
    prepare.add_argument(
        "--representation",
        default="message__all__beto-sts",
    )
    prepare.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    prepare.add_argument("--min-samples", type=int, default=5)
    prepare.add_argument("--percentile", type=int, default=50)
    prepare.add_argument("--model-out", type=Path)

    assign = commands.add_parser(
        "assign",
        help="Vectoriza un mensaje por línea y lo asigna al modelo guardado.",
    )
    assign.add_argument("--input", type=Path, required=True)
    assign.add_argument("--model", type=Path, required=True)
    assign.add_argument("--output", type=Path)
    assign.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    assign.add_argument("--batch-size", type=int, default=32)
    assign.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "prepare":
        model_path = args.model_out or default_model_path(
            args.representation,
            args.min_samples,
            args.percentile,
        )
        artifact = prepare_assigner(
            args.representation,
            args.vector_dir,
            model_path,
            args.min_samples,
            args.percentile,
        )
        print(
            f"Modelo guardado en {model_path}\n"
            f"BETO-STS / coseno / min_samples={artifact['min_samples']} / "
            f"P{artifact['percentile']} / eps={artifact['eps']:.6f}\n"
            f"Train: {artifact['n_train']} instancias, "
            f"{artifact['n_clusters']} clusters, "
            f"ruido={artifact['train_noise_ratio']:.1%}"
        )
        return 0

    output_path = args.output or args.input.with_name(
        f"{args.input.stem}_asignaciones.csv"
    )
    try:
        results = assign_messages(
            args.input,
            args.model,
            output_path,
            args.cache_dir,
            args.batch_size,
            args.device,
        )
    except ImportError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print(
            f"Instala las dependencias con:\n"
            f"  {sys.executable} -m pip install "
            "-r vectorization/requirements.txt",
            file=sys.stderr,
        )
        return 1
    skipped = len(read_messages(args.input)) - len(results)
    artifact = joblib.load(args.model)
    cluster_summary = (
        results.groupby("cluster_assigned", sort=True)
        .agg(
            messages=("input_line", "size"),
            mean_distance_over_eps=("distance_over_eps", "mean"),
        )
        .reset_index()
    )
    train_cluster_sizes = artifact.get("cluster_sizes")
    if train_cluster_sizes is None:
        model = artifact["model"]
        train_cluster_sizes = {
            int(cluster): int(np.count_nonzero(model.labels_ == cluster))
            for cluster in np.unique(model.labels_)
            if cluster >= 0
        }
    cluster_summary["train_cluster_size"] = cluster_summary[
        "cluster_assigned"
    ].map(train_cluster_sizes).fillna(0).astype(int)
    cluster_summary["input_share"] = (
        cluster_summary["messages"] / len(results)
    )
    assigned_ratio = float((~results["is_noise"]).mean())
    print(
        f"Modelo: {artifact['representation']} / "
        f"{artifact['embedding_model']}\n"
        f"DBSCAN: métrica={artifact['metric']}, "
        f"min_samples={artifact['min_samples']}, "
        f"P{artifact['percentile']}, eps={artifact['eps']:.6f}\n"
        f"Entrenamiento: {artifact['n_train']} instancias, "
        f"{artifact['n_clusters']} clusters, "
        f"ruido={artifact['train_noise_ratio']:.1%}\n"
        f"Entrada: {len(results)} mensajes vectorizados, "
        f"{skipped} omitidos por tener menos de {MIN_TOKENS} tokens; "
        f"asignados={assigned_ratio:.1%}, "
        f"ruido={1 - assigned_ratio:.1%}. Resultado: {output_path}"
    )
    print("\nDistribución por cluster:")
    print(cluster_summary.to_string(index=False, formatters={
        "mean_distance_over_eps": lambda value: f"{value:.3f}",
        "input_share": lambda value: f"{value:.1%}",
    }))
    print("\nAsignaciones por mensaje:")
    print(results[[
        "input_line",
        "cluster_assigned",
        "is_noise",
        "nearest_core_cluster",
        "nearest_core_distance",
        "distance_over_eps",
        "assigned_cluster_train_size",
    ]].to_string(index=False, formatters={
        "nearest_core_distance": lambda value: f"{value:.4f}",
        "distance_over_eps": lambda value: f"{value:.3f}",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
