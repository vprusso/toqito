"""Test is_permutation."""
import numpy as np

from toqito.matrix_props import is_permutation

def test_is_permutation():
    """Test if matrix is a real permuattion matrix."""
    mat = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    np.testing.assert_equal(is_permutation(mat), True)


def test_is_not_permutation():
    """Test non-permutation matrix."""
    mat = np.array([[0, 0, 0, 9], [1, 0, 0, 0], [0, 1, 0, 0], [0,0,1,0]])
    np.testing.assert_equal(is_permutation(mat), False)

def test_is_not_permutation():
    """Test non-permutation matrix."""
    mat = np.array([[0, 0, 1,0], [1, 0, 0, 0], [0, 1, 0, 0], [0,0,1,0]])
    np.testing.assert_equal(is_permutation(mat), False)


def test_is_weird_permutation():
    """Test on a weird non-permutation matrix that still adds up to 1."""
    mat = np.array([[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_equal(is_permutation(mat), False)

def test_is_weirder_permutation():
    """Test on an evil non-permutation matrix that still adds up to 1."""
    mat = np.array([[2, -1], [-1, 2]])
    np.testing.assert_equal(is_permutation(mat), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
