"""Test is_identity."""
import numpy as np

from toqito.matrix_props import is_identity


def test_is_hermitian():
    """Test if identity matrix return True."""
    mat = np.eye(4)
    np.testing.assert_equal(is_identity(mat), True)


def test_is_non_identity():
    """Test non-identity matrix."""
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(is_identity(mat), False)


def test_is_identity_not_square():
    """Input must be a square matrix."""
    mat = np.array([[-1, 1, 1], [1, 2, 3]])
    np.testing.assert_equal(is_identity(mat), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
