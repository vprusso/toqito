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
    """Ensure the relative tolerance parameter is honoured."""
    singular_values = np.diag([1.0, 1e-9])
    np.testing.assert_equal(is_rank_one(singular_values, rtol=1e-8), True)


def test_is_rank_one_scale_invariant():
    """Rank detection does not depend on the overall scale of the matrix."""
    # A small scalar multiple of a rank-2 matrix is still rank 2 (the old absolute tolerance reported rank <= 1).
    assert is_rank_one(1e-9 * np.eye(2)) is False
    # A small scalar multiple of a rank-1 matrix is still rank 1.
    assert is_rank_one(1e-9 * np.outer([1, 1], [1, 1])) is True
