"""Test cyclic_permutation."""
import numpy as np

from toqito.matrices import cyclic_permutation_matrix


def test_cyclic_permutation():
    """Test cyclic permuation matrix."""
    n = 10
    res = cyclic_permutation_matrix(n)
    bool_mat = np.allclose(np.linalg.matrix_power(res, n), np.eye(n))
    np.testing.assert_equal(np.all(bool_mat), True)

def test_cyclic_permutation_checks():
    """Run checks to confrim a proper cyclic permutation."""
    for n in (2, 4, 6, 8, 10):
        res = cyclic_permutation_matrix(n)
        # Check the size of the matrix
        assert res.shape == (n , n)
        # Check Cyclic Property: P^n = I
        bool_mat_cyclic = np.allclose(np.linalg.matrix_power(res, n), np.eye(n))
        np.testing.assert_equal(bool_mat_cyclic, True)
        # Check orthogonality: P^T = P^1{-1}
        bool_mat_orthogonal = np.allclose(np.transpose(res), np.linalg.inv(res))
        np.testing.assert_equal(bool_mat_orthogonal, True)

