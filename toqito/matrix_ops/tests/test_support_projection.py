"""Tests for support_projection."""

import numpy as np

from toqito.matrix_ops import support_projection


def test_support_projection_of_zero_matrix_is_zero():
    """support_projection on an all-zero PSD matrix returns the zero matrix."""
    result = support_projection(np.zeros((3, 3)))
    np.testing.assert_allclose(result, np.zeros((3, 3)))


def test_support_projection_of_subtol_eigvals_is_zero():
    """If all eigenvalues are below tol, support_projection returns the zero matrix."""
    mat = 1e-14 * np.eye(2)
    result = support_projection(mat, tol=1e-12)
    np.testing.assert_allclose(result, np.zeros((2, 2)))


def test_support_projection_rank_one():
    """The support projector of a rank-one PSD matrix projects onto its range."""
    mat = np.diag([3.0, 0.0])
    np.testing.assert_allclose(support_projection(mat), np.diag([1.0, 0.0]))


def test_support_projection_full_rank_is_identity():
    """The support projector of a full-rank PSD matrix is the identity."""
    mat = np.diag([2.0, 5.0])
    np.testing.assert_allclose(support_projection(mat), np.eye(2))
