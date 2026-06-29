"""Tests for is_support_subset."""

import numpy as np

from toqito.matrix_props import is_support_subset


def test_is_support_subset_contained():
    """A rank-one support sits inside the full-rank identity."""
    assert is_support_subset(np.diag([1.0, 0.0]), np.eye(2))


def test_is_support_subset_equal_supports():
    """Equal supports are contained in one another."""
    assert is_support_subset(np.eye(2), np.eye(2))


def test_is_support_subset_leaks():
    """A support that extends beyond the other is not contained."""
    assert not is_support_subset(np.eye(2), np.diag([1.0, 0.0]))
