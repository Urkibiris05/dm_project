"""Conservative one-way k-NN relabeling for weak message labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]


@dataclass(frozen=True)
class RelabelRound:
    """Summary of one synchronous propagation round."""

    round_number: int
    changed_this_round: int
    changed_total: int
    remaining_suffer: int
    total_control: int


@dataclass(frozen=True)
class RelabelResult:
    """Final weak labels and their propagation history."""

    labels: IntArray
    history: tuple[RelabelRound, ...]


def _validate_inputs(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.integer],
    subject_ids: NDArray[Any],
    neighbors: int,
    min_control_neighbors: int,
    max_neighbors_per_subject: int | None,
) -> tuple[FloatArray, IntArray, StringArray]:
    vectors = np.asarray(embeddings, dtype=np.float64)
    initial_labels = np.asarray(labels)
    subjects = np.asarray(subject_ids, dtype=object)

    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError("embeddings debe ser una matriz 2D no vacía")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings contiene NaN o valores infinitos")
    if initial_labels.ndim != 1 or initial_labels.shape[0] != vectors.shape[0]:
        raise ValueError("labels debe tener una etiqueta por embedding")
    if not np.isin(initial_labels, [0, 1]).all():
        raise ValueError("labels solo puede contener 0=control y 1=suffer")
    if subjects.ndim != 1 or subjects.shape[0] != vectors.shape[0]:
        raise ValueError("subject_ids debe tener un ID por embedding")
    if any(
        not isinstance(subject, str) or not subject.strip()
        for subject in subjects
    ):
        raise ValueError("subject_ids debe contener cadenas no vacías")
    if not isinstance(neighbors, int) or neighbors < 1:
        raise ValueError("neighbors (J) debe ser un entero >= 1")
    if (
        not isinstance(min_control_neighbors, int)
        or min_control_neighbors < 1
        or min_control_neighbors > neighbors
    ):
        raise ValueError("min_control_neighbors (K) debe cumplir 1 <= K <= J")
    if max_neighbors_per_subject is not None and (
        not isinstance(max_neighbors_per_subject, int)
        or max_neighbors_per_subject < 1
    ):
        raise ValueError("max_neighbors_per_subject debe ser un entero >= 1")

    return vectors, initial_labels.astype(np.int64), subjects.astype(str)


def _normalize_rows(vectors: FloatArray) -> FloatArray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = np.zeros_like(vectors)
    np.divide(vectors, norms, out=normalized, where=norms > 0)
    return normalized


def _nearest_indices(
    normalized: FloatArray,
    query_indices: IntArray,
    subject_ids: StringArray,
    neighbors: int,
    exclude_same_subject: bool,
    max_neighbors_per_subject: int | None,
    block_size: int,
) -> IntArray:
    result = np.empty((query_indices.size, neighbors), dtype=np.int64)

    for start in range(0, query_indices.size, block_size):
        stop = min(start + block_size, query_indices.size)
        query_block = query_indices[start:stop]
        similarities = normalized[query_block] @ normalized.T

        for offset, query_index in enumerate(query_block):
            scores = similarities[offset]
            scores[query_index] = -np.inf
            if exclude_same_subject:
                scores[subject_ids == subject_ids[query_index]] = -np.inf

            valid_indices = np.flatnonzero(np.isfinite(scores))
            if valid_indices.size < neighbors:
                raise ValueError(
                    "No hay suficientes vecinos permitidos para el mensaje "
                    f"de índice {query_index}: se solicitaron {neighbors} y "
                    f"solo hay {valid_indices.size}"
                )
            if max_neighbors_per_subject is None:
                valid_scores = scores[valid_indices]
                cutoff = np.partition(
                    valid_scores, valid_scores.size - neighbors
                )[valid_scores.size - neighbors]
                candidates = valid_indices[valid_scores >= cutoff]
                ranked = candidates[
                    np.lexsort((candidates, -scores[candidates]))
                ]
                selected = ranked[:neighbors].tolist()
            else:
                ranked = valid_indices[
                    np.lexsort((valid_indices, -scores[valid_indices]))
                ]
                selected = []
                subject_counts: dict[object, int] = {}
                for candidate in ranked:
                    candidate_subject = subject_ids[candidate]
                    count = subject_counts.get(candidate_subject, 0)
                    if count >= max_neighbors_per_subject:
                        continue
                    selected.append(int(candidate))
                    subject_counts[candidate_subject] = count + 1
                    if len(selected) == neighbors:
                        break

            if len(selected) < neighbors:
                raise ValueError(
                    "No hay suficientes vecinos permitidos para el mensaje "
                    f"de índice {query_index}: se solicitaron {neighbors} y "
                    f"solo hay {len(selected)}"
                )
            result[start + offset] = selected

    return result


def relabel_messages(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.integer],
    subject_ids: NDArray[Any],
    *,
    neighbors: int,
    min_control_neighbors: int,
    mode: Literal["single_pass", "iterative"] = "single_pass",
    exclude_same_subject: bool = False,
    max_neighbors_per_subject: int | None = None,
    max_rounds: int | None = None,
    block_size: int = 128,
) -> RelabelResult:
    """Relabel suffer messages to control using cosine-neighbor votes.

    Neighbor identities are computed once from the fixed embeddings. Each
    propagation round uses the labels from the previous round, and applies
    all changes synchronously. Control labels are never changed.
    """
    vectors, initial_labels, subjects = _validate_inputs(
        embeddings,
        labels,
        subject_ids,
        neighbors,
        min_control_neighbors,
        max_neighbors_per_subject,
    )
    if mode not in {"single_pass", "iterative"}:
        raise ValueError("mode debe ser 'single_pass' o 'iterative'")
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size debe ser un entero >= 1")
    if max_rounds is not None and (
        not isinstance(max_rounds, int) or max_rounds < 1
    ):
        raise ValueError("max_rounds debe ser un entero >= 1")

    positive_indices = np.flatnonzero(initial_labels == 1)
    if positive_indices.size == 0:
        return RelabelResult(initial_labels.copy(), ())

    normalized = _normalize_rows(vectors)
    neighbor_indices = _nearest_indices(
        normalized,
        positive_indices,
        subjects,
        neighbors,
        exclude_same_subject,
        max_neighbors_per_subject,
        block_size,
    )

    current_labels = initial_labels.copy()
    history: list[RelabelRound] = []
    round_limit = (
        1
        if mode == "single_pass"
        else max_rounds
        if max_rounds is not None
        else positive_indices.size + 1
    )

    for round_number in range(1, round_limit + 1):
        current_positive_rows = np.flatnonzero(
            current_labels[positive_indices] == 1
        )
        current_queries = positive_indices[current_positive_rows]
        current_neighbors = neighbor_indices[current_positive_rows]
        neighbor_labels = current_labels[current_neighbors]
        negative_neighbor_counts = (neighbor_labels == 0).sum(axis=1)
        changed_indices = current_queries[
            negative_neighbor_counts >= min_control_neighbors
        ]
        current_labels[changed_indices] = 0

        changed_total = int(
            np.count_nonzero(initial_labels != current_labels)
        )
        history.append(
            RelabelRound(
                round_number=round_number,
                changed_this_round=int(changed_indices.size),
                changed_total=changed_total,
                remaining_suffer=int(np.count_nonzero(current_labels == 1)),
                total_control=int(np.count_nonzero(current_labels == 0)),
            )
        )
        if changed_indices.size == 0 or mode == "single_pass":
            break

    return RelabelResult(current_labels, tuple(history))
