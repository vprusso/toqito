"""Test is_projection."""
import numpy as np

from toqito.matrix_props import is_projection


def test_is_projection():
    """Check that non-Hermitian projection matrix returns False."""
    mat = np.array([[0, 1], [0, 1]])
    np.testing.assert_equal(is_projection(mat), False)


def test_is_projection_2():
    """Check that projection matrix returns True."""
    mat = np.array([[1, 0], [0, 1]])
    np.testing.assert_equal(is_projection(mat), True)


def test_is_not_pd_non_projection():
    """Check that non-projection matrix returns False."""
    mat = np.array([[-1, -1], [-1, -1]])
    np.testing.assert_equal(is_projection(mat), False)


def test_is_projection_not_square():
    """Input must be a square matrix."""
    mat = np.array([[-1, 1, 1], [1, 2, 3]])
    np.testing.assert_equal(is_projection(mat), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
