"""Tests de las distancias con valores calculados a mano."""

import numpy as np
import pytest

from dbscan_implementation.distances import (
    CosineDistance,
    MinkowskiDistance,
    get_distance,
)

# Diferencias entre ORIGIN y POINT: (3, 4)
ORIGIN = np.array([[0.0, 0.0]])
POINT = np.array([[3.0, 4.0]])


@pytest.mark.parametrize(
    "p, expected",
    [
        (1, 7.0),         # 3 + 4
        (2, 5.0),         # sqrt(9 + 16)
        (np.inf, 4.0),    # max(3, 4)
        (0.5, (np.sqrt(3) + 2) ** 2),  # (sqrt(3) + sqrt(4))^2
    ],
)
def test_minkowski_known_values(p: float, expected: float) -> None:
    distances = MinkowskiDistance(p=p).pairwise(ORIGIN, POINT)
    assert distances.shape == (1, 1)
    assert distances[0, 0] == pytest.approx(expected)


def test_minkowski_pairwise_shape_and_symmetry() -> None:
    X = np.random.default_rng(0).normal(size=(6, 4))
    distances = MinkowskiDistance(p=2).pairwise(X, X)
    assert distances.shape == (6, 6)
    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diag(distances), 0)


def test_minkowski_rejects_invalid_p() -> None:
    with pytest.raises(ValueError):
        MinkowskiDistance(p=0)


def test_cosine_known_values() -> None:
    X = np.array([[1.0, 0.0]])
    others = np.array([
        [5.0, 0.0],    # misma dirección
        [0.0, 2.0],    # ortogonal
        [-3.0, 0.0],   # opuesto
        [0.0, 0.0],    # vector nulo
    ])
    distances = CosineDistance().pairwise(X, others)
    assert distances[0] == pytest.approx([0.0, 1.0, 2.0, 1.0])


def test_cosine_ignores_vector_length() -> None:
    X = np.random.default_rng(0).normal(size=(5, 3))
    distances = CosineDistance().pairwise(X, X)
    scaled = CosineDistance().pairwise(10 * X, 0.1 * X)
    assert np.allclose(distances, scaled)


def test_get_distance_names() -> None:
    assert get_distance("euclidean").p == 2
    assert get_distance("manhattan").p == 1
    assert get_distance("chebyshev").p == np.inf
    assert get_distance("minkowski").p == 2
    assert get_distance("minkowski", p=0.5).p == 0.5
    assert isinstance(get_distance("cosine"), CosineDistance)


def test_get_distance_accepts_object() -> None:
    distance = MinkowskiDistance(p=3)
    assert get_distance(distance) is distance


def test_get_distance_unknown_name() -> None:
    with pytest.raises(ValueError):
        get_distance("no_existe")
