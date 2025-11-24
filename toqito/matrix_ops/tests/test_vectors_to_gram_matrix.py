"""Test vectors_to_gram_matrix."""

import numpy as np
import pytest

from toqito.matrix_ops import vectors_to_gram_matrix
from toqito.states import trine

e_0, e_1 = np.array([[1], [0]]), np.array([[0], [1]])


@pytest.mark.parametrize(
    "vectors, expected_result",
    [
        # Trine states.
        (trine(), np.array([[1, -1 / 2, -1 / 2], [-1 / 2, 1, -1 / 2], [-1 / 2, -1 / 2, 1]])),
        # Orthonormal basis states.
        ([e_0, e_1], np.array([[1, 0], [0, 1]])),
    ],
)
def test_vectors_to_gram_matrix(vectors, expected_result):
    """Test able to construct Gram matrix from vectors."""
    np.testing.assert_allclose(vectors_to_gram_matrix(vectors), expected_result)


@pytest.mark.parametrize(
    "states, expected_result",
    [
        # Orthogonal mixed states
        (
            [
                0.7 * np.array([[1.0, 0.0], [0.0, 0.0]]) + 0.3 * np.eye(2) / 2,
                0.7 * np.array([[0.0, 0.0], [0.0, 1.0]]) + 0.3 * np.eye(2) / 2,
            ],
            np.array([[0.745, 0.255], [0.255, 0.745]]),
        ),
        # Identity matrices
        ([np.eye(2), np.eye(2)], np.array([[2.0, 2.0], [2.0, 2.0]])),
    ],
)
def test_vectors_to_gram_matrix_mixed_states(states, expected_result):
    """Test able to construct Gram matrix from density matrices (mixed states)."""
    np.testing.assert_allclose(vectors_to_gram_matrix(states), expected_result, atol=1e-10)


@pytest.mark.parametrize(
    "vectors",
    [
        # Vectors of different sizes.
        ([np.array([1, 2, 3]), np.array([1, 2])]),
        # Density matrices of different sizes.
        ([np.eye(2), np.eye(3)]),
    ],
)
def test_vectors_to_gram_matrix_invalid_input(vectors):
    """Test function works as expected for an invalid input."""
    with np.testing.assert_raises(ValueError):
        vectors_to_gram_matrix(vectors)
