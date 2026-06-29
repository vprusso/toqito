"""Tests for support_overlap."""

import numpy as np

from toqito.matrix_props import support_overlap


def test_support_overlap_disjoint_is_zero():
    """Matrices with orthogonal supports have zero overlap."""
    assert support_overlap(np.diag([1.0, 0.0]), np.diag([0.0, 1.0])) == 0.0


def test_support_overlap_identical_supports_is_rank():
    """Identical supports overlap by the dimension of the shared subspace."""
    np.testing.assert_allclose(support_overlap(np.eye(3), np.eye(3)), 3.0)


def test_support_overlap_partial():
    """A shared one-dimensional support gives an overlap of one."""
    mat_1 = np.diag([1.0, 1.0, 0.0])
    mat_2 = np.diag([0.0, 1.0, 1.0])
    np.testing.assert_allclose(support_overlap(mat_1, mat_2), 1.0)
