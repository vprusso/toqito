"""Test is_square."""
import numpy as np

from toqito.matrix_props import is_square


def test_is_square():
    """Test that square matrix returns True."""
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(is_square(mat), True)


def test_is_not_square():
    """Test that non-square matrix returns False."""
    mat = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_equal(is_square(mat), False)


def test_is_square_invalid():
    """Input must be a matrix."""
    with np.testing.assert_raises(ValueError):
        is_square(np.array([-1, 1]))


if __name__ == "__main__":
    np.testing.run_module_suite()
