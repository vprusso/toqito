"""Test cyclic_permutation_matrix."""

import numpy as np
import pytest

from toqito.matrices import cyclic_permutation_matrix


@pytest.mark.parametrize("n", [10])
def test_cyclic_permutation_matrix_fixed(n):
    """Test cyclic permuation matrix."""
    res = cyclic_permutation_matrix(n)
    assert np.allclose(np.linalg.matrix_power(res, n), np.eye(n))


@pytest.mark.parametrize(
    "n,k",
    [
        (4, 1),
        (4, 2),
        (4, 3),
    ],
)
def test_cyclic_permutation_matrix_successive(n, k):
    """Test a successive cyclic permuation matrix."""
    res = cyclic_permutation_matrix(n, k)
    assert np.allclose(np.linalg.matrix_power(res, n), np.eye(n))


@pytest.mark.parametrize("n", [2, 4, 6, 8, 10])
def test_cyclic_permutation_matrix_checks(n):
    """Test to confrim a proper cyclic permutation."""
    res = cyclic_permutation_matrix(n)

    # Shape check
    np.testing.assert_equal(res.shape, (n, n))

    # Expected cyclic permutation matrix
    expected_matrix = np.zeros((n, n), dtype=int)
    np.fill_diagonal(expected_matrix[1:], 1)
    expected_matrix[0, -1] = 1

    np.testing.assert_array_equal(res, expected_matrix)


@pytest.mark.parametrize("n", [1.0])
def test_cyclic_permutation_matrix_n_invalid(n):
    """Test function raises TypeError for invalid input 'n'."""
    with pytest.raises(TypeError, match="'n' must be an integer."):
        cyclic_permutation_matrix(n=n)


@pytest.mark.parametrize("n", [-2])
def test_cyclic_permutation_matrix_positive_int(n):
    """Test function raises ValueError for invalid input."""
    with pytest.raises(ValueError, match="'n' must be a positive integer."):
        cyclic_permutation_matrix(n=n)


@pytest.mark.parametrize(
    "n, k",
    [
        (4, 2.0),
    ],
)
def test_cyclic_permutation_matrix_k_invalid(n, k):
    """Test function raises TypeError for invalid input 'k'."""
    with pytest.raises(TypeError, match="'k' must be an integer."):
        cyclic_permutation_matrix(n=n, k=k)
