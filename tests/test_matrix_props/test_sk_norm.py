"""Test S(k) operator norm."""
import numpy as np
import pytest

from toqito.matrix_props import sk_operator_norm
from toqito.states import basis, werner, max_entangled


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

    # See:
    #     N. Johnston. Norms and Cones in the Theory of Quantum Entanglement. PhD thesis (arXiv:1207.1479)
    # Example 5.2.11
    expected = 1 / 8 * (3 + 2 * np.sqrt(2))

    lb, ub = sk_operator_norm(mat)
    np.testing.assert_equal(np.allclose(lb, expected), True)
    np.testing.assert_equal(np.allclose(ub, expected), True)


def test_sk_norm_rank_1():
    """Test S(k) norm of a rank-1 matrix."""
    k = 2
    n = 4
    state = max_entangled(n)
    mat = state @ state.conj().T
    expected = k / n

    lb, ub = sk_operator_norm(mat, k)
    np.testing.assert_equal(np.allclose(lb, expected), True)
    np.testing.assert_equal(np.allclose(ub, expected), True)


@pytest.mark.parametrize("n, a", [(2, 0.5), (2, -0.5), (3, 0.5), (3, -0.5)])
def test_s1_norm_werner(n, a):
    """Test S(1) norm of a Werner state."""
    rho = werner(n, a)
    # See:
    #     N. Johnston. Norms and Cones in the Theory of Quantum Entanglement. PhD thesis (arXiv:1207.1479)
    # Proposition 5.2.10 and Table 5.1
    expected = (1 + abs(min(0, a))) / (n * (n - a))

    lb, ub = sk_operator_norm(rho, k=1)
    np.testing.assert_equal(np.allclose(lb, expected), True)
    np.testing.assert_equal(np.allclose(ub, expected), True)


def test_sk_norm_hermitian_not_psd():
    """Test S(k) norm of a Hermitian but not PSD matrix."""
    e0 = basis(2, 0)
    e00 = np.kron(e0, e0)

    e1 = basis(2, 1)
    e11 = np.kron(e1, e1)

    mat = e00 @ e11.T + e11 @ e00.T
    _, ub = sk_operator_norm(mat)
    np.testing.assert_equal(np.allclose(ub, 1.0), True)


def test_sk_norm_of_zero_matrix():
    """Test S(k) norm of a zero matrix."""
    mat = np.zeros((4, 4))
    lb, ub = sk_operator_norm(mat)
    np.testing.assert_equal(np.allclose(lb, 0.0), True)
    np.testing.assert_equal(np.allclose(ub, 0.0), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
