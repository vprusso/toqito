"""Test is_symmetric."""
import numpy as np

from toqito.matrix_props import is_symmetric


def test_is_symmetric():
    """Test that symmetric matrix returns True."""
    mat = np.array([[1, 7, 3], [7, 4, -5], [3, -5, 6]])
    np.testing.assert_equal(is_symmetric(mat), True)


def test_is_not_symmetric():
    """Test that non-symmetric matrix returns False."""
    mat = np.array([[1, 2], [3, 4]])
    np.testing.assert_equal(is_symmetric(mat), False)


def test_is_symmetric_not_square():
    """Input must be a square matrix."""
    mat = np.array([[-1, 1, 1], [1, 2, 3]])
    np.testing.assert_equal(is_symmetric(mat), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
