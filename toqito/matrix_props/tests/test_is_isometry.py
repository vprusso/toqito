"""Test is_isometry."""

import numpy as np
import pytest

from toqito.matrix_props import is_isometry
from toqito.rand import random_unitary


@pytest.mark.parametrize(
    "mat",
    [
        # A rectangular embedding C^2 -> C^3 with orthonormal columns.
        (np.array([[1, 0], [0, 1], [0, 0]])),
        # The 2x2 identity.
        (np.eye(2)),
        # A Pauli matrix (square isometry = unitary).
        (np.array([[0, 1], [1, 0]])),
    ],
)
def test_is_isometry_true(mat):
    """Matrices with orthonormal columns are isometries."""
    assert is_isometry(mat)


@pytest.mark.parametrize(
    "mat",
    [
        # Columns not orthonormal.
        (np.array([[1, 0], [1, 1]])),
        # Rows orthonormal but columns are not (a co-isometry with dim_out < dim_in).
        (np.array([[1, 0, 0], [0, 1, 0]])),
    ],
)
def test_is_isometry_false(mat):
    """Matrices without orthonormal columns are not isometries."""
    assert not is_isometry(mat)


def test_random_unitary_is_isometry():
    """A random unitary is a square isometry."""
    assert is_isometry(random_unitary(4))


def test_tolerance():
    """A near-isometry is rejected at a tight tolerance and accepted at a loose one."""
    mat = np.array([[1.0, 0.0], [0.0, 1.0 + 1e-4], [0.0, 0.0]])
    assert not is_isometry(mat)
    assert is_isometry(mat, atol=1e-3)
