"""Shared utilities for rq_larra notebooks (Colab/Drive + local).

Reuses the canonical scripts in ``supervised_splits/rama_b_relabeling/``
without modifying them, while fixing two gaps needed for RQ1 and RQ2:
1. Allowing online Hugging Face download fallback when BETO-STS is not yet
   cached locally (e.g. fresh Google Colab runtime).
2. Exporting chronologically ordered message-level logits and probabilities
   (without raw message text) alongside MLP checkpoints and subject-level
   baseline metrics.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


DISORDERS: tuple[str, ...] = ("Ansiedad", "Depresion", "TCA")
DEFAULT_SEEDS: tuple[int, ...] = (11, 23, 37, 51, 79)
BASELINE_CONDITIONS: tuple[str, ...] = (
    "initial_labels",
    "knn_include_same_subject",
)

INDEX_COLUMNS: tuple[str, ...] = (
    "run_id",
    "rq",
    "stage",
    "status",
    "notebook",
    "seed",
    "disorder",
    "embeddings",
    "relabel_neighbors",
    "relabel_min_control",
    "source_partitions",
    "artifacts_dir",
    "notes",
)


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root containing ``particiones/`` and ``rq_larra/``."""
    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start))
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    candidates.extend(
        [
            Path("/content/drive/MyDrive/dm_project"),
            Path("/content/dm_project"),
            Path("/home/larra/dm_project"),
        ]
    )
    for candidate in candidates:
        if (candidate / "particiones").is_dir() and (
            candidate / "supervised_splits" / "rama_b_relabeling"
        ).is_dir():
            return candidate
    raise FileNotFoundError(
        "No se encontró la raíz del proyecto (debe contener 'particiones/' y "
        "'supervised_splits/rama_b_relabeling/'). Clona o monta el repo en Drive."
    )


def ensure_rama_b_on_path(project_root: Path) -> Path:
    """Add ``supervised_splits/rama_b_relabeling`` to ``sys.path``."""
    rama_b_dir = project_root / "supervised_splits" / "rama_b_relabeling"
    if str(rama_b_dir) not in sys.path:
        sys.path.insert(0, str(rama_b_dir))
    return rama_b_dir


def resolve_storage_root(
    project_root: Path,
    drive_storage_dir: str | Path | None = None,
) -> Path:
    """Return persistent directory for heavy .npz embeddings and checkpoints.

    In Google Colab, if ``drive_storage_dir`` is provided (or Drive is mounted
    at ``/content/drive/MyDrive``), heavy binary artifacts are stored on Drive
    so they survive Colab runtime resets without bloating Git.
    """
    if drive_storage_dir is not None:
        root = Path(drive_storage_dir)
    elif Path("/content/drive/MyDrive").is_dir():
        root = Path("/content/drive/MyDrive/dm_project_artifacts")
    else:
        root = project_root / "rq_larra" / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_run_id(tag: str) -> str:
    """Build a collision-safe timestamped run identifier."""
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}"


def register_run(
    project_root: Path,
    *,
    run_id: str,
    rq: str,
    stage: str,
    status: str,
    notebook: str,
    seed: str | int = "",
    disorder: str = "all",
    embeddings: str = "beto_sts",
    relabel_neighbors: int | str = 10,
    relabel_min_control: int | str = 5,
    source_partitions: str = "particiones",
    artifacts_dir: str = "",
    notes: str = "",
) -> Path:
    """Append a run row to ``rq_larra/csv/index_runs.csv``."""
    index_path = project_root / "rq_larra" / "csv" / "index_runs.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not index_path.exists() or index_path.stat().st_size == 0
    row = {
        "run_id": run_id,
        "rq": rq,
        "stage": stage,
        "status": status,
        "notebook": notebook,
        "seed": str(seed),
        "disorder": disorder,
        "embeddings": embeddings,
        "relabel_neighbors": str(relabel_neighbors),
        "relabel_min_control": str(relabel_min_control),
        "source_partitions": source_partitions,
        "artifacts_dir": artifacts_dir,
        "notes": notes,
    }
    with index_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return index_path


def save_latest_baseline_pointer(
    project_root: Path,
    *,
    run_id: str,
    storage_root: Path,
    embedding_dir: Path,
    baseline_model_dir: Path,
    baseline_csv_dir: Path,
    disorders: Iterable[str],
    seeds: Iterable[int],
    relabel_mode: str,
    neighbors: int,
    min_control_neighbors: int,
) -> Path:
    """Persist pointer JSON so rq1.ipynb and rq2.ipynb auto-discover baseline."""
    pointer_path = (
        project_root / "rq_larra" / "csv" / "rq_baseline" / "latest_baseline.json"
    )
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "storage_root": str(storage_root),
        "embedding_dir": str(embedding_dir),
        "baseline_model_dir": str(baseline_model_dir),
        "baseline_csv_dir": str(baseline_csv_dir),
        "disorders": list(disorders),
        "seeds": list(seeds),
        "conditions": list(BASELINE_CONDITIONS),
        "relabel_mode": relabel_mode,
        "neighbors": neighbors,
        "min_control_neighbors": min_control_neighbors,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    pointer_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return pointer_path


def load_latest_baseline_pointer(project_root: Path) -> dict[str, Any]:
    """Load the latest baseline metadata JSON."""
    pointer_path = (
        project_root / "rq_larra" / "csv" / "rq_baseline" / "latest_baseline.json"
    )
    if not pointer_path.is_file():
        raise FileNotFoundError(
            f"No existe {pointer_path}. Ejecuta primero "
            "00_baseline_relabeling.ipynb."
        )
    return json.loads(pointer_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Embeddings, relabeling & baseline MLP wrapper
# ---------------------------------------------------------------------------


def _load_beto_sts_encoder(device: str, torch_threads: int) -> Any:
    """Load pinned BETO-STS encoder, trying local cache first then online."""
    import torch
    from sentence_transformers import SentenceTransformer

    from make_beto_sts_embeddings import MODEL_ID, MODEL_REVISION

    torch.set_num_threads(torch_threads)
    try:
        return SentenceTransformer(
            MODEL_ID,
            device=device,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    except Exception:
        return SentenceTransformer(
            MODEL_ID,
            device=device,
            revision=MODEL_REVISION,
            local_files_only=False,
        )


def ensure_beto_sts_embeddings(
    project_root: Path,
    embedding_dir: Path,
    *,
    partitions: tuple[str, ...] = ("train", "dev"),
    batch_size: int = 32,
    device: str = "cpu",
    torch_threads: int = 4,
) -> Path:
    """Create missing partition .npz embeddings using make_beto_sts_embeddings."""
    ensure_rama_b_on_path(project_root)
    from make_beto_sts_embeddings import encode_partitions

    missing = [p for p in partitions if not (embedding_dir / f"{p}.npz").is_file()]
    if not missing:
        print(f"Usando embeddings BETO-STS en caché: {embedding_dir}")
        return embedding_dir

    encoder = _load_beto_sts_encoder(device=device, torch_threads=torch_threads)
    data_dir = project_root / "particiones"

    if not embedding_dir.exists():
        encode_partitions(
            data_dir=data_dir,
            output_dir=embedding_dir,
            partitions=tuple(missing),
            batch_size=batch_size,
            device=device,
            torch_threads=torch_threads,
            encoder=encoder,
        )
    else:
        tmp_dir = embedding_dir / f"_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        encode_partitions(
            data_dir=data_dir,
            output_dir=tmp_dir,
            partitions=tuple(missing),
            batch_size=batch_size,
            device=device,
            torch_threads=torch_threads,
            encoder=encoder,
        )
        for p in missing:
            (tmp_dir / f"{p}.npz").replace(embedding_dir / f"{p}.npz")
        manifest_path = embedding_dir / "embedding_manifest.json"
        new_manifest = json.loads(
            (tmp_dir / "embedding_manifest.json").read_text(encoding="utf-8")
        )
        if manifest_path.is_file():
            existing_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            existing_manifest.setdefault("splits", {}).update(
                new_manifest.get("splits", {})
            )
            manifest_path.write_text(
                json.dumps(existing_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            (tmp_dir / "embedding_manifest.json").replace(manifest_path)
        tmp_dir.rmdir()

    print(f"Embeddings BETO-STS guardados en: {embedding_dir} ({missing})")
    return embedding_dir


def predict_split_messages_from_checkpoint(
    split_npz: Path,
    checkpoint_npz: Path,
    disorder: str,
) -> pd.DataFrame:
    """Run frozen MLP inference on a split .npz and return chronological logits/probs."""
    import torch
    from torch import nn

    ensure_rama_b_on_path(find_project_root(split_npz))
    from baseline_mlp import _ordered_groups, load_split

    split_data, _ = load_split(split_npz)
    with np.load(checkpoint_npz, allow_pickle=False) as ckpt:
        w0 = ckpt["0_weight"]
        b0 = ckpt["0_bias"]
        w3 = ckpt["3_weight"]
        b3 = ckpt["3_bias"]
        mean = ckpt["scaler_mean"]
        scale = ckpt["scaler_scale"]

    hidden_dim, input_dim = w0.shape
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.0),
        nn.Linear(hidden_dim, 1),
    )
    state = {
        "0.weight": torch.from_numpy(w0.copy()),
        "0.bias": torch.from_numpy(b0.copy()),
        "3.weight": torch.from_numpy(w3.copy()),
        "3.bias": torch.from_numpy(b3.copy()),
    }
    model.load_state_dict(state)
    model.eval()

    x_scaled = torch.from_numpy(
        ((split_data.embeddings - mean) / scale).astype(np.float32)
    )
    with torch.inference_mode():
        logits = model(x_scaled).squeeze(1).numpy().astype(np.float64)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))

    rows: list[pd.DataFrame] = []
    for subject_id, indices in _ordered_groups(split_data, disorder):
        n_msg = len(indices)
        rows.append(
            pd.DataFrame(
                {
                    "disorder": disorder,
                    "subject_id": subject_id,
                    "id_message": split_data.id_message[indices],
                    "date": split_data.date[indices],
                    "message_order": np.arange(1, n_msg + 1, dtype=np.int64),
                    "subject_n_messages": n_msg,
                    "label": split_data.label[indices].astype(np.int64),
                    "logit": logits[indices],
                    "prob": probs[indices],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def run_baseline_and_export(
    project_root: Path,
    *,
    embedding_dir: Path,
    model_output_dir: Path,
    csv_output_dir: Path,
    run_id: str,
    disorders: tuple[str, ...] = DISORDERS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    neighbors: int = 10,
    min_control_neighbors: int = 5,
    relabel_mode: str = "single_pass",
    max_rounds: int | None = None,
    max_epochs: int = 50,
    patience: int = 8,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Run relabeling (include same subject) + baseline MLP and persist all outputs.

    Note: ``knn_exclude_same_subject`` is intentionally omitted per project split.
    """
    ensure_rama_b_on_path(project_root)
    from baseline_mlp import (
        SplitData,
        TrainingConfig,
        _save_checkpoint,
        _train_one,
        _write_relabeling,
        load_split,
    )
    from relabeling import relabel_messages
    from summarize_relabeling import summarize_run

    model_output_dir.mkdir(parents=True, exist_ok=True)
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    train_split, train_model_id = load_split(embedding_dir / "train.npz")
    dev_split, dev_model_id = load_split(embedding_dir / "dev.npz")
    if train_model_id != dev_model_id:
        raise ValueError("Model IDs de train y dev no coinciden.")

    config = TrainingConfig(
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        neighbors=neighbors,
        min_control_neighbors=min_control_neighbors,
    )

    train_csv_path = project_root / "particiones" / "messages_train.csv"
    all_seed_metrics: list[dict[str, Any]] = []
    all_subject_preds: list[pd.DataFrame] = []
    all_msg_preds: list[pd.DataFrame] = []
    relabel_disorder_summaries: list[pd.DataFrame] = []
    relabel_position_summaries: list[pd.DataFrame] = []
    relabel_subject_summaries: list[pd.DataFrame] = []

    for disorder in disorders:
        train_idx = np.flatnonzero(train_split.disorder == disorder)
        dev_idx = np.flatnonzero(dev_split.disorder == disorder)
        train_d = SplitData(
            embeddings=train_split.embeddings[train_idx],
            subject_id=train_split.subject_id[train_idx],
            id_message=train_split.id_message[train_idx],
            disorder=train_split.disorder[train_idx],
            label=train_split.label[train_idx],
            date=train_split.date[train_idx],
        )
        dev_d = SplitData(
            embeddings=dev_split.embeddings[dev_idx],
            subject_id=dev_split.subject_id[dev_idx],
            id_message=dev_split.id_message[dev_idx],
            disorder=dev_split.disorder[dev_idx],
            label=dev_split.label[dev_idx],
            date=dev_split.date[dev_idx],
        )

        relabel_res = relabel_messages(
            train_d.embeddings,
            train_d.label,
            train_d.subject_id,
            neighbors=neighbors,
            min_control_neighbors=min_control_neighbors,
            mode=relabel_mode,
            exclude_same_subject=False,
            max_rounds=max_rounds,
        )

        relabel_dir = csv_output_dir / "relabeling" / disorder
        relabel_dir.mkdir(parents=True, exist_ok=True)
        weak_csv = relabel_dir / "weak_message_labels.csv"
        _write_relabeling(train_split, disorder, relabel_res, weak_csv)
        # Write manifest compatible with summarize_relabeling.py
        relabel_manifest = {
            "model_id": train_model_id,
            "disorder": disorder,
            "neighbors_J": neighbors,
            "min_control_neighbors_K": min_control_neighbors,
            "mode": relabel_mode,
            "exclude_same_subject": False,
            "max_rounds": max_rounds,
            "n_messages": int(len(train_idx)),
            "initial_suffer": int((train_d.label == 1).sum()),
            "initial_control": int((train_d.label == 0).sum()),
            "relabelled_suffer_to_control": int(
                (train_d.label != relabel_res.labels).sum()
            ),
            "final_suffer": int((relabel_res.labels == 1).sum()),
            "final_control": int((relabel_res.labels == 0).sum()),
        }
        (relabel_dir / "run_manifest.json").write_text(
            json.dumps(relabel_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        sums = summarize_run(train_csv_path, relabel_dir, disorder)
        for name, df_sum in sums.items():
            df_sum.to_csv(relabel_dir / f"summary_{name}.csv", index=False)
        relabel_disorder_summaries.append(sums["by_disorder"])
        relabel_position_summaries.append(sums["by_position"])
        relabel_subject_summaries.append(sums["by_subject"])

        condition_targets = (
            ("initial_labels", train_d.label.copy()),
            ("knn_include_same_subject", relabel_res.labels),
        )
        for condition, targets in condition_targets:
            cond_model_dir = model_output_dir / disorder / condition
            cond_model_dir.mkdir(parents=True, exist_ok=True)
            cond_subj_frames: list[pd.DataFrame] = []
            cond_seed_metrics: list[dict[str, Any]] = []

            for seed in seeds:
                ckpt_path = cond_model_dir / f"seed-{seed}.npz"
                model, details = _train_one(
                    train_d.embeddings,
                    targets.astype(np.int64),
                    dev_d.embeddings,
                    dev_d,
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
                cond_subj_frames.append(outcomes)
                all_subject_preds.append(outcomes)
                _save_checkpoint(model, details, ckpt_path)

                msg_df = predict_split_messages_from_checkpoint(
                    embedding_dir / "dev.npz",
                    ckpt_path,
                    disorder,
                )
                msg_df.insert(1, "condition", condition)
                msg_df.insert(2, "seed", seed)
                msg_df["selected_epoch"] = int(details["best_epoch"])
                msg_df["subject_threshold"] = float(details["threshold"])
                all_msg_preds.append(msg_df)

                m_row = {
                    "disorder": disorder,
                    "condition": condition,
                    "model_id": train_model_id,
                    "seed": seed,
                    "train_messages": len(train_idx),
                    "dev_messages": len(dev_idx),
                    "train_subjects": int(np.unique(train_d.subject_id).size),
                    "dev_subjects": int(np.unique(dev_d.subject_id).size),
                    "initial_train_suffer": int(np.sum(train_d.label == 1)),
                    "training_suffer": int(np.sum(targets == 1)),
                    "training_control": int(np.sum(targets == 0)),
                    "selected_epoch": int(details["best_epoch"]),
                    "threshold": float(details["threshold"]),
                    **details["dev_online_metrics"],
                }
                cond_seed_metrics.append(m_row)
                all_seed_metrics.append(m_row)
                print(
                    f"[{disorder} | {condition} | seed={seed}] "
                    f"dev macro-F1={m_row['macro_f1']:.4f} | "
                    f"F1_suffer={m_row['f1_suffer']:.4f} | "
                    f"lat_frac={m_row['latency_tp_fraction_mean']:.3f}"
                )
                del model

            pd.DataFrame(cond_seed_metrics).to_csv(
                cond_model_dir / "seed_metrics.csv", index=False
            )
            pd.concat(cond_subj_frames, ignore_index=True).to_csv(
                cond_model_dir / "dev_subject_predictions.csv", index=False
            )

    pd.concat(relabel_disorder_summaries, ignore_index=True).to_csv(
        csv_output_dir / "relabeling_summary_by_disorder.csv", index=False
    )
    pd.concat(relabel_position_summaries, ignore_index=True).to_csv(
        csv_output_dir / "relabeling_summary_by_position.csv", index=False
    )
    pd.concat(relabel_subject_summaries, ignore_index=True).to_csv(
        csv_output_dir / "relabeling_summary_by_subject.csv", index=False
    )

    metrics_df = pd.DataFrame(all_seed_metrics)
    metrics_df.to_csv(csv_output_dir / "dev_seed_metrics.csv", index=False)

    num_cols = [
        c
        for c in metrics_df.select_dtypes(include="number").columns
        if c != "seed"
    ]
    summary = (
        metrics_df.groupby(["disorder", "condition"], sort=True)[num_cols]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(csv_output_dir / "dev_summary.csv", index=False)

    pd.concat(all_subject_preds, ignore_index=True).to_csv(
        csv_output_dir / "dev_subject_predictions.csv", index=False
    )
    msg_preds_df = pd.concat(all_msg_preds, ignore_index=True)
    msg_preds_df.to_csv(
        csv_output_dir / "dev_message_predictions.csv", index=False
    )

    manifest = {
        "run_id": run_id,
        "status": "completed",
        "model_id": train_model_id,
        "disorders": list(disorders),
        "conditions": list(BASELINE_CONDITIONS),
        "exclude_same_subject_included": False,
        "relabeling": {
            "neighbors_J": neighbors,
            "min_control_neighbors_K": min_control_neighbors,
            "mode": relabel_mode,
            "max_rounds": max_rounds,
            "exclude_same_subject": False,
        },
        "classifier": asdict(config),
        "seeds": list(seeds),
        "test_used": False,
        "message_texts_written_to_outputs": False,
        "embedding_dir": str(embedding_dir),
        "model_output_dir": str(model_output_dir),
        "csv_output_dir": str(csv_output_dir),
    }
    (csv_output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "dev_summary": summary,
        "dev_seed_metrics": metrics_df,
        "dev_message_predictions": msg_preds_df,
    }


def export_frozen_test_predictions(
    *,
    embedding_dir: Path,
    model_output_dir: Path,
    output_csv: Path,
    disorders: tuple[str, ...] = DISORDERS,
    conditions: tuple[str, ...] = BASELINE_CONDITIONS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> pd.DataFrame:
    """Run frozen checkpoint inference on ``test.npz`` (call only when frozen!)."""
    test_npz = embedding_dir / "test.npz"
    if not test_npz.is_file():
        raise FileNotFoundError(
            f"No existe {test_npz}. Genera embeddings de test solo cuando el "
            "protocolo esté congelado."
        )
    frames: list[pd.DataFrame] = []
    for disorder in disorders:
        for condition in conditions:
            for seed in seeds:
                ckpt_path = model_output_dir / disorder / condition / f"seed-{seed}.npz"
                with np.load(ckpt_path, allow_pickle=False) as ckpt:
                    threshold = float(np.asarray(ckpt["threshold"]).item())
                    best_epoch = int(np.asarray(ckpt["best_epoch"]).item())
                df_msg = predict_split_messages_from_checkpoint(
                    test_npz, ckpt_path, disorder
                )
                df_msg.insert(1, "condition", condition)
                df_msg.insert(2, "seed", seed)
                df_msg["selected_epoch"] = best_epoch
                df_msg["subject_threshold"] = threshold
                frames.append(df_msg)
    out = pd.concat(frames, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


# ---------------------------------------------------------------------------
# RQ1 helpers: Confidence, Entropy, Calibration, Selective Classification
# ---------------------------------------------------------------------------


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute Macro-F1, Precision+, Recall+, FPR, FNR, Balanced Acc."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if len(y_true) == 0:
        return {
            "macro_f1": np.nan,
            "precision_pos": np.nan,
            "recall_pos": np.nan,
            "fpr": np.nan,
            "fnr": np.nan,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "roc_auc": np.nan,
        }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else np.nan
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_pos": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_pos": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "fpr": fpr,
        "fnr": fnr,
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))
        if np.unique(y_true).size == 2
        else np.nan,
        "roc_auc": float(roc_auc_score(y_true, y_prob))
        if y_prob is not None and np.unique(y_true).size == 2
        else np.nan,
    }
    return out


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    """Expected Calibration Error (ECE) + bin table for reliability diagrams."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    rows: list[dict[str, float | int]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            rows.append(
                {
                    "bin": i,
                    "lower": lo,
                    "upper": hi,
                    "count": 0,
                    "mean_prob": np.nan,
                    "emp_pos_rate": np.nan,
                    "abs_gap": np.nan,
                }
            )
            continue
        mean_p = float(y_prob[mask].mean())
        emp_acc = float(y_true[mask].mean())
        gap = abs(emp_acc - mean_p)
        ece += (cnt / n) * gap
        rows.append(
            {
                "bin": i,
                "lower": lo,
                "upper": hi,
                "count": cnt,
                "mean_prob": mean_p,
                "emp_pos_rate": emp_acc,
                "abs_gap": gap,
            }
        )
    return float(ece), pd.DataFrame(rows)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Return ECE, Brier score, and NLL (log loss)."""
    y_true_int = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    ece, _ = compute_ece(y_true_int, p, n_bins=n_bins)
    return {
        "ece": ece,
        "brier": float(brier_score_loss(y_true_int, p)),
        "nll": float(log_loss(y_true_int, p, labels=[0, 1])),
    }


def _fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit scalar temperature T > 0 minimizing NLL on calibration logits."""
    from scipy.optimize import minimize_scalar

    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    def nll_fn(log_temp: float) -> float:
        temp = float(np.exp(log_temp))
        scaled = np.clip(logits / temp, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-scaled))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    res = minimize_scalar(nll_fn, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def _apply_temperature(logits: np.ndarray, temp: float) -> np.ndarray:
    scaled = np.clip(np.asarray(logits, dtype=np.float64) / temp, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def fit_subject_oof_calibrators(
    df_group: pd.DataFrame,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute subject-grouped out-of-fold calibrated probabilities on dev.

    Avoids overfitting dev by ensuring no subject appears in both calibration
    fit and evaluation folds. Also returns full-dev calibrators for frozen test.
    """
    df = df_group.copy().reset_index(drop=True)
    y = df["label"].to_numpy(dtype=np.int64)
    logits = df["logit"].to_numpy(dtype=np.float64)
    probs = df["prob"].to_numpy(dtype=np.float64)
    groups = df["subject_id"].to_numpy(dtype=str)

    n_subjects = int(np.unique(groups).size)
    k = min(n_splits, n_subjects)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=2023)

    p_temp = np.zeros_like(probs)
    p_platt = np.zeros_like(probs)
    p_iso = np.zeros_like(probs)

    for tr_idx, va_idx in sgkf.split(logits.reshape(-1, 1), y, groups):
        t_fold = _fit_temperature(logits[tr_idx], y[tr_idx])
        p_temp[va_idx] = _apply_temperature(logits[va_idx], t_fold)

        platt = LogisticRegression(C=1e6, solver="lbfgs")
        platt.fit(logits[tr_idx].reshape(-1, 1), y[tr_idx])
        p_platt[va_idx] = platt.predict_proba(logits[va_idx].reshape(-1, 1))[:, 1]

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(probs[tr_idx], y[tr_idx])
        p_iso[va_idx] = iso.predict(probs[va_idx])

    df["prob_uncal"] = probs
    df["prob_temp"] = p_temp
    df["prob_platt"] = p_platt
    df["prob_isotonic"] = p_iso

    full_temp = _fit_temperature(logits, y)
    full_platt = LogisticRegression(C=1e6, solver="lbfgs").fit(
        logits.reshape(-1, 1), y
    )
    full_iso = IsotonicRegression(out_of_bounds="clip").fit(probs, y)
    full_calibrators = {
        "temperature": full_temp,
        "platt": full_platt,
        "isotonic": full_iso,
    }
    return df, full_calibrators


def compute_entropy_norm(probs: np.ndarray) -> np.ndarray:
    """Normalized binary predictive entropy H_norm(p) in [0, 1]."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return h / np.log(2.0)


def evaluate_selective_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    strategy: str,
    grid: Iterable[float],
    prob_threshold: float = 0.5,
) -> pd.DataFrame:
    """Evaluate confidence or normalized-entropy abstention across a threshold grid."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    conf = np.maximum(y_prob, 1.0 - y_prob)
    ent = compute_entropy_norm(y_prob)
    base_pred = (y_prob >= prob_threshold).astype(np.int64)

    rows: list[dict[str, Any]] = []
    for thr in grid:
        if strategy == "confidence":
            accepted = conf >= float(thr)
        elif strategy == "entropy":
            accepted = ent <= float(thr)
        else:
            raise ValueError(f"Estrategia desconocida: {strategy}")

        n_total = len(y_true)
        n_acc = int(accepted.sum())
        coverage = n_acc / n_total if n_total else 0.0
        if n_acc == 0:
            sel_risk = np.nan
            cls_metrics = compute_binary_metrics(
                np.array([], dtype=np.int64), np.array([], dtype=np.int64)
            )
        else:
            y_t_acc = y_true[accepted]
            y_p_acc = base_pred[accepted]
            sel_risk = float((y_t_acc != y_p_acc).mean())
            cls_metrics = compute_binary_metrics(y_t_acc, y_p_acc, y_prob[accepted])
        rows.append(
            {
                "strategy": strategy,
                "threshold": float(thr),
                "n_total": n_total,
                "n_accepted": n_acc,
                "n_abstained": n_total - n_acc,
                "coverage": coverage,
                "selective_risk": sel_risk,
                **cls_metrics,
            }
        )
    return pd.DataFrame(rows)


def compute_risk_coverage_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    num_points: int = 50,
    prob_threshold: float = 0.5,
) -> tuple[pd.DataFrame, float]:
    """Compute full risk-coverage curve sorted by confidence and AURC."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    conf = np.maximum(y_prob, 1.0 - y_prob)
    pred = (y_prob >= prob_threshold).astype(np.int64)
    errors = (pred != y_true).astype(np.float64)

    order = np.argsort(-conf, kind="mergesort")
    sorted_errors = errors[order]
    n = len(sorted_errors)
    cum_errors = np.cumsum(sorted_errors)
    counts = np.arange(1, n + 1, dtype=np.float64)
    coverages = counts / n
    risks = cum_errors / counts
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    aurc = float(trapz_fn(risks, coverages))

    idx = np.unique(np.linspace(0, n - 1, min(num_points, n), dtype=int))
    curve = pd.DataFrame(
        {
            "coverage": coverages[idx],
            "selective_risk": risks[idx],
            "confidence_cutoff": conf[order][idx],
        }
    )
    return curve, aurc


# ---------------------------------------------------------------------------
# RQ2 helpers: Temporal aggregation, ERDE5/ERDE30 & adaptive alarm models
# ---------------------------------------------------------------------------


def compute_erde(
    labels: np.ndarray,
    predictions: np.ndarray,
    decision_steps: np.ndarray,
    o: int,
) -> float:
    """Official Early Risk Detection Error (ERDE_o) for a user split.

    Parameters
    ----------
    labels:
        True binary user labels (0=control, 1=suffer).
    predictions:
        Predicted binary user alarm decision (1=alarm triggered, 0=no alarm).
    decision_steps:
        1-based index of the message where the alarm fired (ignored if pred==0).
    o:
        Latency penalty inflection parameter (e.g. 5 for ERDE5, 30 for ERDE30).
    """
    y = np.asarray(labels, dtype=np.int64)
    yhat = np.asarray(predictions, dtype=np.int64)
    k = np.asarray(decision_steps, dtype=np.float64)
    n_total = len(y)
    if n_total == 0:
        return float("nan")
    c_fp = float((y == 1).sum()) / n_total
    costs = np.zeros(n_total, dtype=np.float64)
    fp_mask = (yhat == 1) & (y == 0)
    fn_mask = (yhat == 0) & (y == 1)
    tp_mask = (yhat == 1) & (y == 1)
    costs[fp_mask] = c_fp
    costs[fn_mask] = 1.0
    if tp_mask.any():
        costs[tp_mask] = 1.0 - (1.0 / (1.0 + np.exp(np.clip(k[tp_mask] - o, -50.0, 50.0))))
    return float(costs.mean())


def _trailing_slope(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Causal linear regression slope over the last ``window`` points."""
    n = len(values)
    slopes = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        seg = values[max(0, i - window + 1) : i + 1]
        m = len(seg)
        if m < 2:
            slopes[i] = 0.0
        else:
            x = np.arange(m, dtype=np.float64)
            x_c = x - x.mean()
            denom = float(np.dot(x_c, x_c))
            slopes[i] = float(np.dot(x_c, seg - seg.mean()) / denom) if denom > 0 else 0.0
    return slopes


def build_prefix_features_df(
    msg_df: pd.DataFrame,
    *,
    moving_windows: tuple[int, ...] = (3, 5, 10),
    ewma_lambdas: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7),
    slope_window: int = 5,
) -> pd.DataFrame:
    """Compute strictly causal prefix features per subject in chronological order."""
    df = msg_df.sort_values(
        ["subject_id", "message_order"], kind="stable"
    ).reset_index(drop=True)
    blocks: list[pd.DataFrame] = []
    for _, grp in df.groupby("subject_id", sort=False):
        p = grp["prob"].to_numpy(dtype=np.float64)
        n = len(p)
        counts = np.arange(1, n + 1, dtype=np.float64)
        mean_cum = np.cumsum(p) / counts
        max_cum = np.maximum.accumulate(p)
        # Causal cumulative std (ddof=0)
        mean_sq_cum = np.cumsum(p * p) / counts
        std_cum = np.sqrt(np.maximum(0.0, mean_sq_cum - mean_cum * mean_cum))

        out = grp.copy()
        out["mean_cum"] = mean_cum
        out["max_cum"] = max_cum
        out["std_cum"] = std_cum
        out["slope_recent"] = _trailing_slope(p, window=slope_window)
        out["prop_above_05"] = np.cumsum((p >= 0.5).astype(np.float64)) / counts

        for k in moving_windows:
            win_means = np.array(
                [p[max(0, i - k + 1) : i + 1].mean() for i in range(n)],
                dtype=np.float64,
            )
            out[f"recent_mean_{k}"] = win_means

        out["diff_recent5_hist"] = out.get(
            "recent_mean_5",
            np.array([p[max(0, i - 4) : i + 1].mean() for i in range(n)]),
        ) - mean_cum

        for lam in ewma_lambdas:
            ewma = np.empty(n, dtype=np.float64)
            ewma[0] = p[0]
            for i in range(1, n):
                ewma[i] = lam * p[i] + (1.0 - lam) * ewma[i - 1]
            tag = str(lam).replace(".", "")
            out[f"ewma_{tag}"] = ewma

        blocks.append(out)
    return pd.concat(blocks, ignore_index=True)


def evaluate_alarm_rule(
    prefix_df: pd.DataFrame,
    *,
    score_col: str,
    threshold: float,
    consecutive_r: int = 1,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simulate causal irreversible alarm triggering across subjects."""
    rows: list[dict[str, Any]] = []
    for subject_id, grp in prefix_df.groupby("subject_id", sort=False):
        scores = grp[score_col].to_numpy(dtype=np.float64)
        n = len(scores)
        label = int(grp["label"].iloc[0])
        triggered_at: int | None = None

        if consecutive_r <= 1:
            hits = np.flatnonzero(scores >= threshold)
            if hits.size > 0:
                triggered_at = int(hits[0] + 1)
        else:
            streak = 0
            for idx, val in enumerate(scores):
                if val >= threshold:
                    streak += 1
                    if streak >= consecutive_r:
                        triggered_at = idx + 1
                        break
                else:
                    streak = 0

        alarm = int(triggered_at is not None)
        decision_msg = triggered_at if triggered_at is not None else n
        rows.append(
            {
                "subject_id": subject_id,
                "label": label,
                "prediction": alarm,
                "alarm_triggered": bool(alarm),
                "decision_at_message": int(decision_msg),
                "decision_fraction": float(decision_msg / n),
                "n_messages": int(n),
                "score_at_decision": float(scores[decision_msg - 1]),
                "max_score": float(scores.max()),
            }
        )

    outcomes = pd.DataFrame(rows)
    y = outcomes["label"].to_numpy(dtype=np.int64)
    pred = outcomes["prediction"].to_numpy(dtype=np.int64)
    steps = outcomes["decision_at_message"].to_numpy(dtype=np.int64)

    cls = compute_binary_metrics(
        y, pred, outcomes["max_score"].to_numpy(dtype=np.float64)
    )
    tp_rows = outcomes.loc[(outcomes["label"] == 1) & (outcomes["prediction"] == 1)]
    alarm_rows = outcomes.loc[outcomes["alarm_triggered"]]
    metrics: dict[str, float] = {
        **cls,
        "erde5": compute_erde(y, pred, steps, o=5),
        "erde30": compute_erde(y, pred, steps, o=30),
        "tp_latency_msg_mean": float(tp_rows["decision_at_message"].mean())
        if not tp_rows.empty
        else np.nan,
        "tp_latency_msg_median": float(tp_rows["decision_at_message"].median())
        if not tp_rows.empty
        else np.nan,
        "tp_latency_frac_mean": float(tp_rows["decision_fraction"].mean())
        if not tp_rows.empty
        else np.nan,
        "pct_users_alarmed": float(outcomes["alarm_triggered"].mean()),
        "pct_users_no_alarm": float((~outcomes["alarm_triggered"]).mean()),
        "alarm_latency_msg_mean_all": float(alarm_rows["decision_at_message"].mean())
        if not alarm_rows.empty
        else np.nan,
    }
    return outcomes, metrics


def fit_predict_oof_temporal_alarm(
    prefix_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    model_type: str = "logistic",
    n_splits: int = 5,
) -> tuple[np.ndarray, Any]:
    """Fit Logistic or Ridge alarm model via subject-grouped CV on dev prefixes.

    Weights each prefix inversely by its subject's total message count so users
    with very long histories do not dominate the temporal decision boundary.
    """
    df = prefix_df.copy().reset_index(drop=True)
    x = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    groups = df["subject_id"].to_numpy(dtype=str)
    weights = 1.0 / df["subject_n_messages"].to_numpy(dtype=np.float64)

    n_subjects = int(np.unique(groups).size)
    k = min(n_splits, n_subjects)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=2023)
    oof_scores = np.zeros(len(df), dtype=np.float64)

    def _make_est() -> tuple[StandardScaler, Any]:
        scaler = StandardScaler()
        if model_type == "logistic":
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=2023)
        elif model_type == "ridge":
            clf = Ridge(alpha=1.0, random_state=2023)
        else:
            raise ValueError(f"model_type no soportado: {model_type}")
        return scaler, clf

    for tr_idx, va_idx in sgkf.split(x, y, groups):
        scaler, est = _make_est()
        x_tr = scaler.fit_transform(x[tr_idx])
        x_va = scaler.transform(x[va_idx])
        est.fit(x_tr, y[tr_idx], sample_weight=weights[tr_idx])
        if model_type == "logistic":
            oof_scores[va_idx] = est.predict_proba(x_va)[:, 1]
        else:
            oof_scores[va_idx] = est.predict(x_va)

    full_scaler, full_est = _make_est()
    x_full = full_scaler.fit_transform(x)
    full_est.fit(x_full, y, sample_weight=weights)
    return oof_scores, {"scaler": full_scaler, "estimator": full_est}
