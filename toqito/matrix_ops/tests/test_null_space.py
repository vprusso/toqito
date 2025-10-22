"""Tests for null_space."""

import numpy as np
import pytest

from toqito.matrix_ops import null_space


def test_null_space_simple_matrix():
    """Null space basis should annihilate the input matrix."""
    mat = np.array([[1, 1, 0], [0, 0, 0]], dtype=float)
    basis = null_space(mat)
    np.testing.assert_allclose(
        mat @ basis,
        np.zeros((2, basis.shape[1])),
        atol=1e-12,
    )


def test_null_space_empty_matrix_returns_empty_basis():
    """Zero-dimensional matrices yield an empty null space basis."""
    mat = np.zeros((0, 0), dtype=float)
    basis = null_space(mat)
    assert basis.shape == (0, 0)


def test_null_space_full_rank_returns_empty():
    """Full-rank matrices have a zero-dimensional null space."""
    mat = np.eye(3)
    basis = null_space(mat)
    assert basis.shape == (3, 0)


def test_null_space_invalid_dimension_raises():
    """Reject one-dimensional inputs that are not matrices."""
    with pytest.raises(ValueError):
        null_space(np.array([1, 2, 3]))
