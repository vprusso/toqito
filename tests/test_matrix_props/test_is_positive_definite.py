"""Test is_positive_definite."""
import numpy as np

from toqito.matrix_props import is_positive_definite


def test_is_is_positive_definite():
    """Check that positive definite matrix returns True."""
    mat = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    np.testing.assert_equal(is_positive_definite(mat), True)


def test_is_not_positive_definite():
    """Check that non-positive definite matrix returns False."""
    mat = np.array([[-1, -1], [-1, -1]])
    np.testing.assert_equal(is_positive_definite(mat), False)


def test_is_positive_definite_not_hermitian():
    """Input must be a Hermitian matrix."""
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(is_positive_definite(mat), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
