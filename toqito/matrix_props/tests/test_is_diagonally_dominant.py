"""Test is_diagonally_dominant."""

import numpy as np

from toqito.matrix_props import is_diagonally_dominant


def test_is_not_square():
    """Test that non-square matrix returns False."""
    mat = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_equal(is_diagonally_dominant(mat), False)


def test_diagonally_dominant():
    """Check that strict diagonally dominance.

    Matrix examples from :footcite:`WikiDiagDom`.
    """
    # Diagonally dominant (but not strict)
    mat = np.array([[3, -2, 1], [1, 3, 2], [-1, 2, 4]])
    np.testing.assert_equal(is_diagonally_dominant(mat, is_strict=False), True)

    # Non-diagonally dominant
    mat = np.array([[-2, 2, 1], [1, 3, 2], [1, -2, 0]])
    np.testing.assert_equal(is_diagonally_dominant(mat), False)

    # Strictly diagonally dominant
    mat = np.array([[-4, 2, 1], [1, 6, 2], [1, -2, 5]])
    np.testing.assert_equal(is_diagonally_dominant(mat, is_strict=True), True)


def test_is_not_diagonally_dominant():
    """Check that diagonally dominant matrix returns False."""
    mat = np.array([[-1, 2], [-1, -1]])
    np.testing.assert_equal(is_diagonally_dominant(mat), False)


def test_strict_diagonally_dominant():
    """Check that strict diagonally dominant matrix returns False."""
    mat = np.array([[-1, 1], [-1, -1]])
    np.testing.assert_equal(is_diagonally_dominant(mat), False)
