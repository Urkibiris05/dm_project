"""Tests for weak-label k-NN relabeling."""

from __future__ import annotations

import numpy as np
import pytest

from relabeling import relabel_messages


def test_single_pass_relabels_suffer_with_enough_control_neighbors() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ]
    )
    labels = np.array([1, 0, 1])
    subjects = np.array(["suffer-a", "control-a", "suffer-b"])

    result = relabel_messages(
        embeddings,
        labels,
        subjects,
        neighbors=1,
        min_control_neighbors=1,
    )

    np.testing.assert_array_equal(result.labels, [0, 0, 0])
    assert len(result.history) == 1
    assert result.history[0].changed_this_round == 2
    assert result.history[0].changed_total == 2


def test_query_does_not_vote_for_itself() -> None:
    embeddings = np.array([[1.0, 0.0], [-1.0, 0.0]])
    labels = np.array([1, 1])
    subjects = np.array(["a", "b"])

    result = relabel_messages(
        embeddings,
        labels,
        subjects,
        neighbors=1,
        min_control_neighbors=1,
    )

    np.testing.assert_array_equal(result.labels, labels)
    assert result.history[0].changed_this_round == 0


def test_control_labels_never_change() -> None:
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
    labels = np.array([0, 1, 1])
    subjects = np.array(["control", "positive-a", "positive-b"])

    result = relabel_messages(
        embeddings,
        labels,
        subjects,
        neighbors=1,
        min_control_neighbors=1,
    )

    assert result.labels[0] == 0
    assert set(result.labels).issubset({0, 1})


def test_same_subject_exclusion_applies_before_neighbor_selection() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.1],
            [0.8, 0.6],
        ]
    )
    labels = np.array([1, 1, 0])
    subjects = np.array(["same", "same", "control"])

    result = relabel_messages(
        embeddings,
        labels,
        subjects,
        neighbors=1,
        min_control_neighbors=1,
        exclude_same_subject=True,
    )

    np.testing.assert_array_equal(result.labels, [0, 0, 0])


def test_subject_neighbor_cap_is_respected() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.1],
            [0.95, 0.3],
        ]
    )
    labels = np.array([1, 0, 0])
    subjects = np.array(["positive", "control-a", "control-a"])

    with pytest.raises(ValueError, match="suficientes vecinos"):
        relabel_messages(
            embeddings,
            labels,
            subjects,
            neighbors=2,
            min_control_neighbors=1,
            max_neighbors_per_subject=1,
        )


@pytest.mark.parametrize(
    ("neighbors", "min_control_neighbors"),
    [(0, 1), (2, 0), (1, 2)],
)
def test_invalid_j_k_is_rejected(
    neighbors: int,
    min_control_neighbors: int,
) -> None:
    with pytest.raises(ValueError):
        relabel_messages(
            np.eye(2),
            np.array([1, 0]),
            np.array(["a", "b"]),
            neighbors=neighbors,
            min_control_neighbors=min_control_neighbors,
        )


def test_iterative_mode_records_stopping_round() -> None:
    result = relabel_messages(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([1, 0]),
        np.array(["positive", "control"]),
        neighbors=1,
        min_control_neighbors=1,
        mode="iterative",
    )

    assert [entry.changed_this_round for entry in result.history] == [1, 0]
    assert result.history[-1].remaining_suffer == 0


def test_iterative_mode_propagates_changes_in_later_rounds() -> None:
    angle_10 = np.deg2rad(10)
    angle_minus_20 = np.deg2rad(-20)
    embeddings = np.array(
        [
            [1.0, 0.0],
            [np.cos(angle_minus_20), np.sin(angle_minus_20)],
            [np.cos(angle_10), np.sin(angle_10)],
        ]
    )
    labels = np.array([1, 1, 0])
    subjects = np.array(["positive-a", "positive-b", "control"])

    result = relabel_messages(
        embeddings,
        labels,
        subjects,
        neighbors=1,
        min_control_neighbors=1,
        mode="iterative",
    )

    np.testing.assert_array_equal(result.labels, [0, 0, 0])
    assert [entry.changed_this_round for entry in result.history] == [1, 1, 0]


def test_equal_similarity_ties_are_broken_by_input_order() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    labels = np.array([1, 0, 1])
    subjects = np.array(["query", "control-first", "positive-second"])

    result = relabel_messages(
        embeddings,
        labels,
        subjects,
        neighbors=1,
        min_control_neighbors=1,
    )

    assert result.labels[0] == 0
