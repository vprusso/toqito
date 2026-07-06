"""Test is_positive_semidefinite."""

import numpy as np

from toqito.matrix_props import is_positive_semidefinite


def test_is_positive_semidefinite():
    """Test that positive semidefinite matrix returns True."""
    mat = np.array([[1, -1], [-1, 1]])
    np.testing.assert_equal(is_positive_semidefinite(mat), True)


def test_is_not_positive_semidefinite():
    """Test that non-positive semidefinite matrix returns False."""
    mat = np.array([[-1, -1], [-1, -1]])
    np.testing.assert_equal(is_positive_semidefinite(mat), False)


def test_is_positive_semidefinite_uses_relative_tolerance():
    """Test that the eigenvalue cutoff honors relative tolerance."""
    mat = np.diag([1e6, -2e-2])
    np.testing.assert_equal(is_positive_semidefinite(mat, rtol=1e-8, atol=1e-8), False)
    np.testing.assert_equal(is_positive_semidefinite(mat, rtol=3e-8, atol=1e-8), True)
