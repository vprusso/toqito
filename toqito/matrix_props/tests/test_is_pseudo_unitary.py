"""Test is_pseudo_unitary."""

import numpy as np
import pytest

from toqito.matrix_props import is_pseudo_unitary
from toqito.rand import random_unitary


@pytest.mark.parametrize(
    "mat, p, q, expected",
    [
        (random_unitary(2), 2, 0, True),  # Unitary Matrix
        (np.array([[np.cosh(0.5), np.sinh(0.5)], [np.sinh(0.5), np.cosh(0.5)]]), 1, 1, True),  # Lorentz Boost Matrix
        (np.array([[1, 0], [1, -1]]), 1, 1, False),  # Non pseudo unitary matrix
        (np.array([[1, 0], [1, 1]]), 4, 5, False),  # Inconsistent shapes of matrix and signature
        (np.array([[-1, 1, 1], [1, 2, 3]]), 1, 1, False),  # Non square matrix
    ],
)
def test_is_pseudo_unitary(mat, p, q, expected):
    """Test that is_pseudo_unitary gives correct boolean value on valid inputs."""
    np.testing.assert_equal(is_pseudo_unitary(mat, p, q), expected)


def test_is_pseudo_unitary_value_error():
    """Input must have p >= 0 and q >= 0."""
    mat = np.array([[1, 0], [0, 1]])
    np.testing.assert_raises_regex(ValueError, "p and q must be non-negative", is_pseudo_unitary, mat, p=-1, q=1)
    np.testing.assert_raises_regex(ValueError, "p and q must be non-negative", is_pseudo_unitary, mat, p=1, q=-1)
