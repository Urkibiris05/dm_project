"""Train and evaluate the weak-label BETO-STS + MLP baseline on dev."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from relabeling import RelabelResult, relabel_messages


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_EMBEDDING_DIR = (
    MODULE_DIR / "reports" / "embeddings" / "beto_sts_full_baseline"
)
DEFAULT_OUTPUT_ROOT = MODULE_DIR / "reports" / "baseline"
DISORDERS = ("Ansiedad", "Depresion", "TCA")
CONDITIONS = (
    "initial_labels",
    "knn_include_same_subject",
    "knn_exclude_same_subject",
)


@dataclass(frozen=True)
class TrainingConfig:
    """Provisional small MLP configuration for the baseline."""

    hidden_dim: int = 256
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 50
    patience: int = 8
    neighbors: int = 10
    min_control_neighbors: int = 5


@dataclass(frozen=True)
class SplitData:
    """Message-level vectors and metadata loaded from a text-free NPZ."""

    embeddings: np.ndarray
    subject_id: np.ndarray
    id_message: np.ndarray
    disorder: np.ndarray
    label: np.ndarray
    date: np.ndarray


def load_split(path: Path) -> tuple[SplitData, str]:
    """Load one split and reject missing, invalid, or misaligned arrays."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "embeddings",
                "subject_id",
                "id_message",
                "disorder",
                "label",
                "date",
                "partition",
                "model_id",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"Faltan arrays: {missing}")
            data = SplitData(
                embeddings=np.asarray(archive["embeddings"], dtype=np.float32),
                subject_id=np.asarray(archive["subject_id"]).astype(str),
                id_message=np.asarray(archive["id_message"]).astype(str),
                disorder=np.asarray(archive["disorder"]).astype(str),
                label=np.asarray(archive["label"], dtype=np.int64),
                date=np.asarray(archive["date"]).astype(str),
            )
            partition = str(np.asarray(archive["partition"]).item())
            model_id = str(np.asarray(archive["model_id"]).item())
    except (OSError, KeyError) as exc:
        raise ValueError(f"No se pudo leer la caché {path}: {exc}") from exc

    if partition != path.stem:
        raise ValueError(
            f"La caché {path} declara partición {partition}"
        )
    n_rows = data.embeddings.shape[0] if data.embeddings.ndim == 2 else -1
    arrays = (
        data.subject_id,
        data.id_message,
        data.disorder,
        data.label,
        data.date,
    )
    if n_rows < 1 or any(len(array) != n_rows for array in arrays):
        raise ValueError(f"Arrays de {path} con dimensiones desalineadas")
    if not np.isfinite(data.embeddings).all():
        raise ValueError(f"Embeddings no finitos en {path}")
    if not np.isin(data.label, [0, 1]).all():
        raise ValueError(f"Etiquetas no binarias en {path}")
    frame = pd.DataFrame(
        {
            "disorder": data.disorder,
            "subject_id": data.subject_id,
            "id_message": data.id_message,
        }
    )
    if frame.duplicated().any():
        raise ValueError(f"Claves de mensaje duplicadas en {path}")
    return data, model_id


def _ordered_groups(
    data: SplitData,
    disorder: str,
) -> list[tuple[str, np.ndarray]]:
    """Return subject message indices ordered by date, then message ID."""
    mask = data.disorder == disorder
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        raise ValueError(f"No hay mensajes para {disorder}")
    dates = pd.to_datetime(data.date[positions], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Fechas inválidas en {disorder}")
    order_frame = pd.DataFrame(
        {
            "position": positions,
            "subject_id": data.subject_id[positions],
            "date": dates,
            "id_message": data.id_message[positions],
        }
    ).sort_values(
        ["subject_id", "date", "id_message"],
        kind="stable",
    )
    groups = [
        (str(subject_id), group["position"].to_numpy(dtype=np.int64))
        for subject_id, group in order_frame.groupby(
            "subject_id",
            sort=False,
        )
    ]
    for _, indices in groups:
        if np.unique(data.label[indices]).size != 1:
            raise ValueError(
                f"Un sujeto tiene etiquetas discordantes para {disorder}"
            )
    return groups


def subject_max_prefix_scores(
    probabilities: np.ndarray,
    data: SplitData,
    disorder: str,
) -> pd.DataFrame:
    """Summarize each subject by the maximum cumulative mean seen online."""
    if probabilities.ndim != 1 or len(probabilities) != len(data.label):
        raise ValueError("Debe haber una probabilidad por mensaje")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilidades no finitas")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Las probabilidades deben estar entre 0 y 1")

    rows: list[dict[str, Any]] = []
    for subject_id, indices in _ordered_groups(data, disorder):
        prefix_means = np.cumsum(probabilities[indices]) / np.arange(
            1,
            len(indices) + 1,
        )
        rows.append(
            {
                "subject_id": subject_id,
                "label": int(data.label[indices[0]]),
                "max_prefix_mean": float(prefix_means.max()),
                "n_messages": int(len(indices)),
                "prefix_means": prefix_means,
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(
    subject_scores: pd.DataFrame,
) -> tuple[float, dict[str, float]]:
    """Choose a dev threshold by subject-level macro-F1."""
    labels = subject_scores["label"].to_numpy(dtype=np.int64)
    scores = subject_scores["max_prefix_mean"].to_numpy(dtype=np.float64)
    if np.unique(labels).size != 2:
        raise ValueError("Dev debe contener ambas clases para elegir umbral")
    unique_scores = np.unique(scores)
    candidates = np.unique(
        np.concatenate(
            [
                unique_scores,
                [np.nextafter(unique_scores[0], -np.inf)],
                [np.nextafter(unique_scores[-1], np.inf)],
                [0.5],
            ]
        )
    )
    best: tuple[float, float, float] | None = None
    best_threshold = 0.5
    best_predictions = np.zeros_like(labels)
    for threshold in candidates:
        predictions = (scores >= threshold).astype(np.int64)
        macro_f1 = float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        )
        candidate_key = (
            macro_f1,
            -abs(float(threshold) - 0.5),
            float(threshold),
        )
        if best is None or candidate_key > best:
            best = candidate_key
            best_threshold = float(threshold)
            best_predictions = predictions
    return best_threshold, classification_metrics(labels, best_predictions)


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute subject-level binary classification metrics."""
    metrics = {
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "f1_suffer": float(
            f1_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "precision_suffer": float(
            precision_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "recall_suffer": float(
            recall_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
    }
    if scores is not None and np.unique(labels).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def evaluate_online(
    probabilities: np.ndarray,
    data: SplitData,
    disorder: str,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Apply irreversible threshold crossing to cumulative mean scores."""
    subject_scores = subject_max_prefix_scores(
        probabilities,
        data,
        disorder,
    )
    predictions = (subject_scores["max_prefix_mean"] >= threshold).astype(
        np.int64
    )
    subjects = _ordered_groups(data, disorder)
    latency_rows: list[dict[str, Any]] = []
    for (subject_id, indices), prediction in zip(subjects, predictions):
        prefix_means = np.cumsum(probabilities[indices]) / np.arange(
            1,
            len(indices) + 1,
        )
        crossings = np.flatnonzero(prefix_means >= threshold)
        decision_at = (
            int(crossings[0] + 1) if crossings.size else int(len(indices))
        )
        latency_rows.append(
            {
                "subject_id": subject_id,
                "label": int(data.label[indices[0]]),
                "prediction": int(prediction),
                "max_prefix_mean": float(prefix_means.max()),
                "n_messages": int(len(indices)),
                "decision_at_message": decision_at,
                "decision_fraction": decision_at / len(indices),
                "crossed_threshold": bool(crossings.size),
            }
        )
    outcomes = pd.DataFrame(latency_rows)
    labels = outcomes["label"].to_numpy(dtype=np.int64)
    predicted = outcomes["prediction"].to_numpy(dtype=np.int64)
    metrics = classification_metrics(
        labels,
        predicted,
        outcomes["max_prefix_mean"].to_numpy(dtype=np.float64),
    )
    true_positive_detections = outcomes.loc[
        outcomes["label"].eq(1) & outcomes["prediction"].eq(1)
    ]
    positive_subjects = int(outcomes["label"].eq(1).sum())
    metrics["latency_tp_messages_mean"] = (
        float(true_positive_detections["decision_at_message"].mean())
        if not true_positive_detections.empty
        else float("nan")
    )
    metrics["latency_tp_fraction_mean"] = (
        float(true_positive_detections["decision_fraction"].mean())
        if not true_positive_detections.empty
        else float("nan")
    )
    metrics["positive_subject_detection_coverage"] = (
        len(true_positive_detections) / positive_subjects
        if positive_subjects
        else float("nan")
    )
    return outcomes, metrics


def _train_one(
    train_vectors: np.ndarray,
    train_labels: np.ndarray,
    dev_vectors: np.ndarray,
    dev_data: SplitData,
    disorder: str,
    seed: int,
    config: TrainingConfig,
) -> tuple[Any, dict[str, Any]]:
    """Train one MLP seed and select epoch/threshold on dev only."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "El entrenamiento del MLP requiere torch. Usa el venv del "
            "proyecto o instala vectorization/requirements.txt."
        ) from exc

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    mean = train_vectors.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_vectors.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    x_train = torch.from_numpy((train_vectors - mean) / scale)
    x_dev = torch.from_numpy((dev_vectors - mean) / scale)
    y_train = torch.from_numpy(train_labels.astype(np.float32))

    model = nn.Sequential(
        nn.Linear(train_vectors.shape[1], config.hidden_dim),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, 1),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    best_macro_f1 = -1.0
    best_epoch = 0
    best_threshold = 0.5
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(x_train), generator=rng)
        for start in range(0, len(x_train), config.batch_size):
            batch_indices = permutation[start:start + config.batch_size]
            logits = model(x_train[batch_indices]).squeeze(1)
            loss = criterion(logits, y_train[batch_indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            dev_logits = model(x_dev).squeeze(1)
            dev_probabilities = torch.sigmoid(dev_logits).numpy()
        scores = subject_max_prefix_scores(
            dev_probabilities,
            dev_data,
            disorder,
        )
        threshold, metrics = choose_threshold(scores)
        macro_f1 = metrics["macro_f1"]
        if macro_f1 > best_macro_f1 + 1e-12:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_threshold = threshold
            best_metrics = metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("No se guardó un checkpoint de desarrollo")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        final_dev_probabilities = torch.sigmoid(
            model(x_dev).squeeze(1)
        ).numpy()
    outcomes, online_metrics = evaluate_online(
        final_dev_probabilities,
        dev_data,
        disorder,
        best_threshold,
    )
    details: dict[str, Any] = {
        "seed": seed,
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "dev_selection_macro_f1": best_macro_f1,
        "dev_selection_metrics": best_metrics,
        "dev_online_metrics": online_metrics,
        "dev_outcomes": outcomes,
        "scaler_mean": mean,
        "scaler_scale": scale,
        "dev_probabilities": final_dev_probabilities,
    }
    return model, details


def _save_checkpoint(
    model: Any,
    details: dict[str, Any],
    path: Path,
) -> None:
    arrays: dict[str, np.ndarray] = {
        key.replace(".", "_"): value.detach().cpu().numpy()
        for key, value in model.state_dict().items()
    }
    arrays["scaler_mean"] = details["scaler_mean"]
    arrays["scaler_scale"] = details["scaler_scale"]
    arrays["threshold"] = np.array(details["threshold"], dtype=np.float64)
    arrays["best_epoch"] = np.array(details["best_epoch"], dtype=np.int64)
    np.savez_compressed(path, **arrays)


def _fit_condition(
    *,
    train: SplitData,
    dev: SplitData,
    disorder: str,
    labels: np.ndarray,
    condition: str,
    run_dir: Path,
    model_id: str,
    seeds: tuple[int, ...],
    config: TrainingConfig,
) -> list[dict[str, Any]]:
    train_mask = train.disorder == disorder
    dev_mask = dev.disorder == disorder
    train_indices = np.flatnonzero(train_mask)
    dev_indices = np.flatnonzero(dev_mask)
    train_data = SplitData(
        embeddings=train.embeddings[train_indices],
        subject_id=train.subject_id[train_indices],
        id_message=train.id_message[train_indices],
        disorder=train.disorder[train_indices],
        label=train.label[train_indices],
        date=train.date[train_indices],
    )
    dev_data = SplitData(
        embeddings=dev.embeddings[dev_indices],
        subject_id=dev.subject_id[dev_indices],
        id_message=dev.id_message[dev_indices],
        disorder=dev.disorder[dev_indices],
        label=dev.label[dev_indices],
        date=dev.date[dev_indices],
    )
    target_labels = np.asarray(labels[train_indices], dtype=np.int64)
    condition_dir = run_dir / disorder / condition
    condition_dir.mkdir(parents=True, exist_ok=False)

    all_metrics: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for seed in seeds:
        model, details = _train_one(
            train_data.embeddings,
            target_labels,
            dev_data.embeddings,
            dev_data,
            disorder,
            seed,
            config,
        )
        outcomes: pd.DataFrame = details.pop("dev_outcomes")
        details.pop("dev_probabilities")
        outcomes["disorder"] = disorder
        outcomes["condition"] = condition
        outcomes["seed"] = seed
        outcomes["threshold"] = details["threshold"]
        prediction_frames.append(outcomes)
        _save_checkpoint(
            model,
            details,
            condition_dir / f"seed-{seed}.npz",
        )
        metrics = {
            "disorder": disorder,
            "condition": condition,
            "model_id": model_id,
            "seed": seed,
            "train_messages": len(train_indices),
            "dev_messages": len(dev_indices),
            "train_subjects": int(
                np.unique(train_data.subject_id).size
            ),
            "dev_subjects": int(np.unique(dev_data.subject_id).size),
            "initial_train_suffer": int(
                np.sum(train_data.label == 1)
            ),
            "training_suffer": int(np.sum(target_labels == 1)),
            "training_control": int(np.sum(target_labels == 0)),
            "selected_epoch": int(details["best_epoch"]),
            "threshold": float(details["threshold"]),
            **details["dev_online_metrics"],
        }
        all_metrics.append(metrics)
        print(
            f"{disorder} {condition} seed={seed}: "
            f"dev macro-F1={metrics['macro_f1']:.3f}, "
            f"F1_suffer={metrics['f1_suffer']:.3f}, "
            f"latencyTP={metrics['latency_tp_fraction_mean']:.3f}"
        )
        del model

    pd.DataFrame(all_metrics).to_csv(
        condition_dir / "seed_metrics.csv",
        index=False,
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        condition_dir / "dev_subject_predictions.csv",
        index=False,
    )
    return all_metrics


def _write_relabeling(
    train: SplitData,
    disorder: str,
    result: RelabelResult,
    path: Path,
) -> None:
    indices = np.flatnonzero(train.disorder == disorder)
    labels = train.label[indices]
    pd.DataFrame(
        {
            "subject_id": train.subject_id[indices],
            "id_message": train.id_message[indices],
            "initial_label": labels,
            "weak_label": result.labels,
            "was_relabelled": labels != result.labels,
        }
    ).to_csv(path, index=False)
    pd.DataFrame(
        [asdict(round_summary) for round_summary in result.history]
    ).to_csv(
        path.with_name(f"round_history_{path.stem}.csv"),
        index=False,
    )


def run_baseline(
    *,
    embedding_dir: Path = DEFAULT_EMBEDDING_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_name: str,
    seeds: tuple[int, ...] = (11, 23, 37, 51, 79),
    config: TrainingConfig = TrainingConfig(),
    disorders: tuple[str, ...] = DISORDERS,
) -> pd.DataFrame:
    """Run MLP-Init and k-NN MLP conditions, selecting on dev only."""
    if not run_name or any(char not in "abcdefghijklmnopqrstuvwxyz"
                           "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                           for char in run_name):
        raise ValueError("run_name solo puede contener letras, números, . _ -")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Se requiere una lista de semillas única y no vacía")
    if not disorders or set(disorders) - set(DISORDERS):
        raise ValueError(f"Trastornos inválidos: {disorders}")
    train, train_model_id = load_split(embedding_dir / "train.npz")
    dev, dev_model_id = load_split(embedding_dir / "dev.npz")
    if train_model_id != dev_model_id:
        raise ValueError("Los modelos de train y dev no coinciden")
    if str(np.unique(train.disorder).tolist()) != str(
        np.unique(dev.disorder).tolist()
    ):
        raise ValueError("Los trastornos disponibles en train y dev difieren")

    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "model_id": train_model_id,
        "training_labels": {
            "initial_labels": "subject labels copied weakly to messages",
            "knn_include_same_subject": (
                "single-pass weak labels; same-subject neighbors allowed "
                "except the query itself"
            ),
            "knn_exclude_same_subject": "single-pass weak labels; all "
            "same-subject neighbors excluded",
        },
        "knn": {
            "neighbors_J": config.neighbors,
            "min_control_neighbors_K": config.min_control_neighbors,
            "metric": "cosine",
            "mode": "single_pass",
            "query_message_excluded": True,
            "exclude_same_subject_condition": True,
        },
        "classifier": {
            "architecture": (
                "Linear(d,256)->ReLU->Dropout(0.3)->Linear(256,1)"
            ),
            "loss": "BCEWithLogitsLoss",
            "class_weights": None,
            "optimizer": "AdamW",
            "config": asdict(config),
        },
        "online_decision": (
            "mean cumulative message probability; irreversible positive "
            "decision at first threshold crossing; threshold selected on dev"
        ),
        "selection_metric": "subject-level dev macro-F1",
        "seeds": list(seeds),
        "disorders": list(disorders),
        "test_used": False,
        "message_texts_written_to_outputs": False,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    all_metrics: list[dict[str, Any]] = []
    for disorder in disorders:
        train_indices = np.flatnonzero(train.disorder == disorder)
        disorder_labels = train.label[train_indices]
        disorder_subjects = train.subject_id[train_indices]
        vectors = train.embeddings[train_indices]

        include_result = relabel_messages(
            vectors,
            disorder_labels,
            disorder_subjects,
            neighbors=config.neighbors,
            min_control_neighbors=config.min_control_neighbors,
            mode="single_pass",
            exclude_same_subject=False,
        )
        exclude_result = relabel_messages(
            vectors,
            disorder_labels,
            disorder_subjects,
            neighbors=config.neighbors,
            min_control_neighbors=config.min_control_neighbors,
            mode="single_pass",
            exclude_same_subject=True,
        )
        disorder_dir = output_dir / disorder
        disorder_dir.mkdir()
        _write_relabeling(
            train,
            disorder,
            include_result,
            disorder_dir / "weak_labels_include_same_subject.csv",
        )
        _write_relabeling(
            train,
            disorder,
            exclude_result,
            disorder_dir / "weak_labels_exclude_same_subject.csv",
        )
        condition_labels = (
            ("initial_labels", disorder_labels.copy()),
            ("knn_include_same_subject", include_result.labels),
            ("knn_exclude_same_subject", exclude_result.labels),
        )
        for condition, targets in condition_labels:
            target_labels_full = train.label.copy()
            target_labels_full[train_indices] = targets
            all_metrics.extend(
                _fit_condition(
                    train=train,
                    dev=dev,
                    disorder=disorder,
                    labels=target_labels_full,
                    condition=condition,
                    run_dir=output_dir,
                    model_id=train_model_id,
                    seeds=seeds,
                    config=config,
                )
            )

    metrics_frame = pd.DataFrame(all_metrics)
    metrics_frame.to_csv(output_dir / "dev_seed_metrics.csv", index=False)
    numeric_columns = [
        column
        for column in metrics_frame.select_dtypes(include="number").columns
        if column != "seed"
    ]
    summary = (
        metrics_frame.groupby(
            ["disorder", "condition"],
            sort=True,
        )[numeric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in summary.columns
    ]
    summary = summary.reset_index()
    summary.to_csv(output_dir / "dev_summary.csv", index=False)
    manifest["status"] = "completed"
    manifest["summary_csv"] = str(output_dir / "dev_summary.csv")
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nDevelopment summary:")
    print(summary.to_string(index=False))
    print(f"\nOutputs: {output_dir}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_DIR,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11, 23, 37, 51, 79],
    )
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--min-control-neighbors", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    config = TrainingConfig(
        neighbors=args.neighbors,
        min_control_neighbors=args.min_control_neighbors,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    run_baseline(
        embedding_dir=args.embedding_dir,
        output_root=args.output_root,
        run_name=args.run_name,
        seeds=tuple(args.seeds),
        config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
