"""Test S(k) operator norm."""

import numpy as np
import pytest

from toqito.matrix_props import sk_operator_norm
from toqito.states import basis, max_entangled, werner


def test_s1_norm_example():
    """Test S(1) norm of a density matrix."""
    mat = (
        np.array(
            [
                [5, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        )
        / 8
    )

    # See Example 5.2.11 of :cite:`Johnston_2012_Norms`
    expected = 1 / 8 * (3 + 2 * np.sqrt(2))

    lower_bound, upper_bound = sk_operator_norm(mat)
    np.testing.assert_equal(np.allclose(lower_bound, expected, atol=0.001), True)
    np.testing.assert_equal(np.allclose(upper_bound, expected, atol=0.001), True)


def test_sk_norm_rank_1():
    """Test S(k) norm of a rank-1 matrix."""
    k = 2
    dim = 4
    state = max_entangled(dim)
    mat = state @ state.conj().T
    expected = k / dim

    lower_bound, upper_bound = sk_operator_norm(mat, k)
    np.testing.assert_equal(np.allclose(lower_bound, expected), True)
    np.testing.assert_equal(np.allclose(upper_bound, expected), True)


@pytest.mark.parametrize("n, a", [(2, 0.5), (2, -0.5), (3, 0.5), (3, -0.5)])
def test_s1_norm_werner(n, a):
    """Test S(1) norm of a Werner state."""
    rho = werner(n, a)
    # See:
    # N. Johnston.
    # Norms and Cones in the Theory of Quantum Entanglement.
    # PhD thesis (arXiv:1207.1479)
    # Proposition 5.2.10 and Table 5.1
    expected = (1 + abs(min(0, a))) / (n * (n - a))

    lower_bound, upper_bound = sk_operator_norm(rho, k=1)
    np.testing.assert_equal(np.allclose(lower_bound, expected, atol=1e-4), True)
    np.testing.assert_equal(np.allclose(upper_bound, expected, atol=1e-4), True)


def test_sk_norm_hermitian_not_psd():
    """Test S(k) norm of a Hermitian but not PSD matrix."""
    e_0 = basis(2, 0)
    e_00 = np.kron(e_0, e_0)

    e_1 = basis(2, 1)
    e_11 = np.kron(e_1, e_1)

    mat = e_00 @ e_11.T + e_11 @ e_00.T
    _, upper_bound = sk_operator_norm(mat)
    np.testing.assert_equal(np.allclose(upper_bound, 1.0), True)


def test_sk_norm_of_zero_matrix():
    """Test S(k) norm of a zero matrix."""
    mat = np.zeros((4, 4))
    lower_bound, upper_bound = sk_operator_norm(mat)
    np.testing.assert_equal(np.allclose(lower_bound, 0.0), True)
    np.testing.assert_equal(np.allclose(upper_bound, 0.0), True)
