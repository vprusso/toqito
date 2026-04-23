"""Test S(k) operator norm."""

import numpy as np
import pytest

from toqito.matrix_props import sk_operator_norm
from toqito.matrix_props.sk_norm import __lower_bound_sk_norm_randomized
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

    # See Example 5.2.11 of :footcite:`Johnston_2012_Norms`
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


def test_sk_norm_k_larger_than_dim():
    """Test S(k) norm when k is larger than the dimensions."""
    dim = 3
    k = 4  # k is larger than the matrix dimension
    mat = max_entangled(dim) @ max_entangled(dim).conj().T

    lower_bound, upper_bound = sk_operator_norm(mat, k)
    expected = np.linalg.norm(mat, ord=2)
    np.testing.assert_equal(np.allclose(lower_bound, expected), True)
    np.testing.assert_equal(np.allclose(upper_bound, expected), True)


def test_sk_norm_non_square_matrix():
    """Test S(k) norm with a non-square matrix (should raise an error)."""
    mat = np.random.rand(4, 5)  # Non-square matrix
    with pytest.raises(ValueError, match="Input matrix must be square."):
        sk_operator_norm(mat, k=2)


def test_sk_norm_zero_rank_matrix():
    """Test S(k) norm of a zero-rank matrix."""
    mat = np.zeros((4, 4))
    lower_bound, upper_bound = sk_operator_norm(mat)
    np.testing.assert_equal(np.allclose(lower_bound, 0.0), True)
    np.testing.assert_equal(np.allclose(upper_bound, 0.0), True)


def test_sk_norm_hermitian_psd():
    """Test S(k) norm of a Hermitian positive semi-definite matrix."""
    mat = np.array([[1, 0], [0, 1]])  # Identity matrix
    lower_bound, upper_bound = sk_operator_norm(mat, k=1)
    np.testing.assert_equal(np.allclose(lower_bound, 1.0), True)
    np.testing.assert_equal(np.allclose(upper_bound, 1.0), True)


def test_sk_norm_randomized_bound():
    """Test randomized method for lower bound of the S(k)-norm."""
    mat = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    mat = mat @ mat.T.conj()  # Make the matrix Hermitian
    lower_bound, upper_bound = sk_operator_norm(mat, k=2)
    assert lower_bound <= upper_bound


def test_sk_norm_target_met():
    """Test early exit when target value is met."""
    mat = np.eye(4)
    target = 1.0  # Target value that will be met early
    lower_bound, upper_bound = sk_operator_norm(mat, k=1, target=target)
    np.testing.assert_equal(np.allclose(lower_bound, target), True)
    np.testing.assert_equal(np.allclose(upper_bound, target), True)


def test_sk_norm_non_hermitian_matrix():
    """Test S(k) norm of a non-Hermitian matrix."""
    mat = np.array([[1, 2], [3, 4]])  # Non-Hermitian matrix
    lower_bound, upper_bound = sk_operator_norm(mat, k=1)

    # Assert that the lower bound and upper bound are non-negative
    assert lower_bound >= 0
    assert upper_bound >= lower_bound


def test_sk_operator_norm_scalar_dim_must_divide():
    """A scalar `dim` that does not evenly divide the matrix size is rejected."""
    with pytest.raises(ValueError, match="evenly divide"):
        sk_operator_norm(np.eye(6), k=1, dim=4)


def test_sk_operator_norm_non_hermitian_returns_basic_bounds():
    """A non-Hermitian bipartite matrix falls through to the basic k/min(dim) lower bound."""
    # Non-Hermitian 4x4 matrix with rank > 1 on a 2x2 bipartite system.
    mat = np.array(
        [
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 5.0],
            [0.0, 0.0, 0.0, 6.0],
        ]
    )
    lower, upper = sk_operator_norm(mat, k=1, dim=[2, 2])
    assert lower >= 0
    assert upper >= lower


def test_sk_operator_norm_k_equals_two_exercises_k_gt_1_sdp_branch():
    """k=2 on a bipartite qutrit state exercises the k>1 partial-trace SDP constraint branch."""
    rho = max_entangled(3) @ max_entangled(3).conj().T
    rho /= np.trace(rho)
    lower, upper = sk_operator_norm(rho, k=2, dim=[3, 3], effort=1)
    assert 0 <= lower
    assert lower <= upper


def test_sk_norm_randomized_start_vec_schmidt_rank_warns_and_falls_back():
    """Private helper: `start_vec` with Schmidt rank > k emits a warning and falls back to random init."""
    dim = [2, 2]
    max_ent = max_entangled(2, is_normalized=False)
    result = np.zeros(1)
    with pytest.warns(UserWarning, match="Schmidt rank"):
        result = __lower_bound_sk_norm_randomized(np.eye(4), k=1, dim=dim, tol=1e-4, start_vec=max_ent.reshape(-1))
    assert np.isfinite(result)


def test_sk_norm_randomized_start_vec_low_schmidt_rank_used_as_init():
    """Private helper: `start_vec` with Schmidt rank ≤ k is used directly as the iteration seed."""
    dim = [2, 2]
    product = np.kron(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    result = __lower_bound_sk_norm_randomized(np.eye(4), k=1, dim=dim, tol=1e-4, start_vec=product)
    assert np.isfinite(result)
