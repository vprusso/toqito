"""Test is_positive_semidefinite."""
import numpy as np

from toqito.matrix_props import is_positive_semidefinite


def test_is_positive_semidefinite():
    """Test that positive semidefinite matrix returns True."""
    mat = np.array([[1, -1], [-1, 1]])
    np.testing.assert_equal(is_positive_semidefinite(mat), True)


def test_is_not_positive_semidefinite():
    """Test that non-positive semidefinite matrix returns False."""
    mat = np.array([[-1, -1], [-1, -1]])
    np.testing.assert_equal(is_positive_semidefinite(mat), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
