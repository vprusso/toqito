"""Test is_diagonal."""
import numpy as np

from toqito.matrix_props import is_diagonal


def test_is_diagonal():
    """Test if matrix is diagonal."""
    mat = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    np.testing.assert_equal(is_diagonal(mat), True)


def test_is_non_diagonal():
    """Test non-diagonal matrix."""
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(is_diagonal(mat), False)


def test_is_diagonal_non_square():
    """Test on a non-square matrix."""
    mat = np.array([[1, 0, 0], [0, 1, 0]])
    np.testing.assert_equal(is_diagonal(mat), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
