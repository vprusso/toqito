"""Test is_permutation."""
import numpy as np

from toqito.matrix_props.is_permutation import is_permutation

def test_is_simple_permutation():
    """Test if matrix is a simple permutation matrix."""
    mat = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    np.testing.assert_equal(is_permutation(mat), True)

def test_is_not_permutation():
    """Test non-permutation matrix wih one element being 9 !in set (0,1)."""
    mat = np.array([[0, 0, 0, 9], [1, 0, 0, 0], [0, 1, 0, 0], [0,0,1,0]])
    np.testing.assert_equal(is_permutation(mat), False)

def test_is_not_permutation_also():
    """Test non-permutation matrix where the row and column check will fail."""
    mat = np.array([[0, 0, 1,0], [1, 0, 0, 0], [0, 1, 0, 0], [0,0,1,0]])
    np.testing.assert_equal(is_permutation(mat), False)

def test_is_weird_not_permutation():
    """Test on a weird non-permutation matrix that still adds up to 1."""
    mat = np.array([[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_equal(is_permutation(mat), False)

def test_is_insidious_not_permutation():
    """Test on an insidious non-permutation matrix that still adds up to 1."""
    mat = np.array([[2, -1], [-1, 2]])
    np.testing.assert_equal(is_permutation(mat), False)

if __name__ == "__main__":
    np.testing.run_module_suite()
    
