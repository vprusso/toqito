"""Test is_rank_one."""

import numpy as np
import pytest

from toqito.matrix_props import is_rank_one


@pytest.mark.parametrize(
    "matrix",
    [np.array([[2], [3]], dtype=np.complex128) @ np.array([[2, 3]], dtype=np.complex128)],
)
def test_rank_one_matrix_returns_true(matrix):
    """Check that rank-one matrices are identified correctly."""
    np.testing.assert_equal(is_rank_one(matrix), True)


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(2, dtype=np.complex128),
        np.array([[1, 0], [0, 1e-6]], dtype=np.complex128),
    ],
)
def test_full_rank_matrix_returns_false(matrix):
    """Check that higher-rank matrices are rejected."""
    np.testing.assert_equal(is_rank_one(matrix), False)


def test_small_singular_values_respected_by_tolerance():
    """Ensure the tolerance parameter is honoured."""
    singular_values = np.diag([1.0, 1e-9])
    np.testing.assert_equal(is_rank_one(singular_values, tol=1e-8), True)
