"""Test is_commuting."""
import numpy as np

from toqito.matrix_props import is_commuting


def test_is_commuting_false():
    """Test if non-commuting matrices return False."""
    mat_1 = np.array([[0, 1], [0, 0]])
    mat_2 = np.array([[1, 0], [0, 0]])
    np.testing.assert_equal(is_commuting(mat_1, mat_2), False)


def test_is_commuting_true():
    """Test commuting matrices return True."""
    mat_1 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 2]])
    mat_2 = np.array([[2, 4, 0], [3, 1, 0], [-1, -4, 1]])
    np.testing.assert_equal(is_commuting(mat_1, mat_2), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
