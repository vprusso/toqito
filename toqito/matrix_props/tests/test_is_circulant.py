"""Test is_circulant."""

import numpy as np

from toqito.matrix_props import is_circulant


def test_is_circulant_false():
    """Test if non-circulant matrices return False."""
    mat = np.array([[0, 1], [0, 0]])
    np.testing.assert_equal(is_circulant(mat), False)


def test_is_circulant_non_square():
    """Test if non-circulant matrices return False."""
    mat = np.array([[0, 1, 0], [0, 0, 1]])
    np.testing.assert_equal(is_circulant(mat), False)


def test_is_circulant_true():
    """Test circulant matrices return True."""
    mat = np.array([[4, 1, 2, 3], [3, 4, 1, 2], [2, 3, 4, 1], [1, 2, 3, 4]])
    np.testing.assert_equal(is_circulant(mat), True)
