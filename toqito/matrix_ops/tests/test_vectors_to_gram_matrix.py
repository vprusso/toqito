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
    ],
)
def test_vectors_to_gram_matrix(vectors, expected_result):
    """Test able to construct Gram matrix from vectors."""
    np.testing.assert_allclose(vectors_to_gram_matrix(vectors), expected_result)


@pytest.mark.parametrize(
    "vectors",
    [
        # Vectors of different sizes.
        ([np.array([1, 2, 3]), np.array([1, 2])]),
    ],
)
def test_vectors_to_gram_matrix_invalid_input(vectors):
    """Test function works as expected for an invalid input."""
    with np.testing.assert_raises(ValueError):
        vectors_to_gram_matrix(vectors)
