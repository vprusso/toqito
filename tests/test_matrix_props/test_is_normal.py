"""Test is_normal."""
import numpy as np

from toqito.matrix_props import is_normal


def test_is_normal():
    """Test that normal matrix returns True."""
    mat = np.identity(4)
    np.testing.assert_equal(is_normal(mat), True)


def test_is_not_normal():
    """Test that non-normal matrix returns False."""
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(is_normal(mat), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
