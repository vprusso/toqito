"""Test is_pd."""
import numpy as np

from toqito.matrix_props import is_pd


def test_is_pd():
    """Check that positive definite matrix returns True."""
    mat = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    np.testing.assert_equal(is_pd(mat), True)


def test_is_not_pd():
    """Check that non-positive definite matrix returns False."""
    mat = np.array([[-1, -1], [-1, -1]])
    np.testing.assert_equal(is_pd(mat), False)


def test_is_not_pd2():
    """Check that non-square matrix returns False."""
    mat = np.array([[1, 2, 3], [2, 1, 4]])
    np.testing.assert_equal(is_pd(mat), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
