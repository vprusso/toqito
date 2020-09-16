"""Test is_unitary."""
import numpy as np

from toqito.matrix_props import is_unitary
from toqito.random import random_unitary


def test_is_unitary_random():
    """Test that random unitary matrix returns True."""
    mat = random_unitary(2)
    np.testing.assert_equal(is_unitary(mat), True)


def test_is_unitary_hardcoded():
    """Test that hardcoded unitary matrix returns True."""
    mat = np.array([[0, 1], [1, 0]])
    np.testing.assert_equal(is_unitary(mat), True)


def test_is_not_unitary():
    """Test that non-unitary matrix returns False."""
    mat = np.array([[1, 0], [1, 1]])
    np.testing.assert_equal(is_unitary(mat), False)


def test_is_not_unitary_matrix():
    """Test that non-unitary matrix returns False."""
    mat = np.array([[1, 0], [1, 1]])
    np.testing.assert_equal(is_unitary(mat), False)


def test_is_unitary_not_square():
    """Input must be a square matrix."""
    mat = np.array([[-1, 1, 1], [1, 2, 3]])
    np.testing.assert_equal(is_unitary(mat), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
