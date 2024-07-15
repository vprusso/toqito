"""Test cyclic_permutation_matrix."""

import numpy as np

from toqito.matrices import cyclic_permutation_matrix


def test_cyclic_permutation_matrix_fixed():
    """Test cyclic permuation matrix."""
    n = 10
    res = cyclic_permutation_matrix(n)
    assert np.allclose(np.linalg.matrix_power(res, n), np.eye(n))


def test_cyclic_permutation_matrix_successive():
    """Test a successive cyclic permuation matrix."""
    n = 4
    # Test for k from 1 to n - 1
    for k in range(1, n):
        res = cyclic_permutation_matrix(n, k)
        assert np.allclose(np.linalg.matrix_power(res, n), np.eye(n))


def test_cyclic_permutation_matrix_checks():
    """Run checks to confrim a proper cyclic permutation."""
    for n in (2, 4, 6, 8, 10):
        res = cyclic_permutation_matrix(n)
        # Check the size of the matrix
        np.testing.assert_equal(res.shape, (n, n))
        # Check that the matrix is equal to the calculated matirx
        expected_matrix = np.zeros((n, n), dtype=int)
        np.fill_diagonal(expected_matrix[1:], 1)
        expected_matrix[0, -1] = 1

        np.testing.assert_array_equal(res, expected_matrix)
