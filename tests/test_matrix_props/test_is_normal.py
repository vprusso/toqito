"""Test is_normal."""
import numpy as np

from toqito.matrix_props import is_normal


def test_is_normal():
    """Check that normal and non-unitary and non-Hermitian matrix yields True.

    Normal matrix obtained from:
    https://en.wikipedia.org/wiki/Normal_matrix
    """
    mat = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    np.testing.assert_equal(is_normal(mat), True)


def test_is_normal_identity():
    """Test that the identity matrix returns True."""
    mat = np.identity(4)
    np.testing.assert_equal(is_normal(mat), True)


def test_is_not_normal():
    """Test that non-normal matrix returns False."""
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(is_normal(mat), False)


def test_is_normal_not_square():
    """Input must be a square matrix."""
    mat = np.array([[-1, 1, 1], [1, 2, 3]])
    np.testing.assert_equal(is_normal(mat), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
