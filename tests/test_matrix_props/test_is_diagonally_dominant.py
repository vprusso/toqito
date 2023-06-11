"""Test is_diagonally_dominant."""
import numpy as np

from toqito.matrix_props import is_diagonally_dominant


def test_is_diagonally_dominant():
    """Check that diagonally dominant matrix returns True."""
    mat = np.array([[2, -1, 0], [0, 2, -1], [0, -1, 2]])
    np.testing.assert_equal(is_diagonally_dominant(mat), True)


def test_is_not_diagonally_dominant():
    """Check that diagonally dominant matrix returns False."""
    mat = np.array([[-1, 2], [-1, -1]])
    np.testing.assert_equal(is_diagonally_dominant(mat), False)


def test_strict_diagonally_dominant():
    """Check that strict diagonally dominant matrix returns False."""
    mat = np.array([[-1, 1], [-1, -1]])
    np.testing.assert_equal(is_diagonally_dominant(mat), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
