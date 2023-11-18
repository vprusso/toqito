"""Test is_hermitian."""
import numpy as np

from toqito.matrix_props import is_hermitian


def test_is_hermitian():
    """Test if matrix is Hermitian."""
    mat = np.array([[2, 2 + 1j, 4], [2 - 1j, 3, 1j], [4, -1j, 1]])
    np.testing.assert_equal(is_hermitian(mat), True)


def test_is_non_hermitian():
    """Test non-Hermitian matrix."""
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(is_hermitian(mat), False)


def test_is_hermitian_not_square():
    """Input must be a square matrix."""
    mat = np.array([[-1, 1, 1], [1, 2, 3]])
    np.testing.assert_equal(is_hermitian(mat), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
