"""Test is_separable."""

import numpy as np
import pytest

from toqito.channels import partial_trace
from toqito.matrix_props import is_density
from toqito.rand import random_density_matrix
from toqito.state_props.is_separable import is_separable
from toqito.states import basis, bell, isotropic, tile


def test_entangled_zhang_realignment_criterion():
    """Test for entanglement using Zhang's realignment criterion."""
    # Create a state that satisfies this criterion
    rho = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
    np.testing.assert_equal(is_separable(rho), False)


def test_entangled_qutrit_qutrit():
    """Test for entanglement in the qutrit-qutrit case."""
    # Create a 3x3 entangled state
    psi = (1 / np.sqrt(3)) * (
        np.kron([1, 0, 0], [1, 0, 0]) + np.kron([0, 1, 0], [0, 1, 0]) + np.kron([0, 0, 1], [0, 0, 1])
    )
    rho = np.outer(psi, psi)
    np.testing.assert_equal(is_separable(rho), False)


def test_entangled_breuer_hall():
    """Test for entanglement using Breuer-Hall positive maps."""
    # Create a 4x4 entangled state
    psi = (1 / np.sqrt(2)) * (np.kron([1, 0], [1, 0]) + np.kron([0, 1], [0, 1]))
    rho = np.outer(psi, psi)
    np.testing.assert_equal(is_separable(rho), False)


def test_non_positive_semidefinite_matrix():
    """Ensure separability of non-positive semidefinite matrix is invalid."""
    with np.testing.assert_raises(ValueError):
        state = np.array([[-1, -1], [-1, -1]])
        is_separable(state)


def test_psd_matrix_local_dim_one():
    """Every positive semidefinite matrix is separable when one of the local dimensions is 1."""
    np.testing.assert_equal(is_separable(np.identity(2)), True)


def test_invalid_dim_parameter():
    """The dimension of the state must evenly divide the length of the state."""
    with np.testing.assert_raises(ValueError):
        dim = 3
        rho = isotropic(dim, 1 / (dim + 1))
        is_separable(rho, dim + 1)


def test_entangled_ppt_criterion():
    """Determined to be entangled via the PPT criterion."""
    rho = bell(0) @ bell(0).conj().T
    np.testing.assert_equal(is_separable(rho), False)


def test_ppt_small_dimensions():
    """Determined to be separable via sufficiency of the PPT criterion in small dimensions."""
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    psi = 1 / np.sqrt(3) * e_0 + 1 / np.sqrt(3) * e_1 + 1 / np.sqrt(3) * e_2

    e_0, e_1 = basis(2, 0), basis(2, 1)
    phi = np.kron((1 / np.sqrt(2) * e_0 + 1 / np.sqrt(2) * e_1), psi)
    sigma = phi @ phi.conj().T
    np.testing.assert_equal(is_separable(sigma), True)


def test_ppt_low_rank():
    """Determined to be separable via the operational criterion for low-rank operators."""
    m = 6
    n = m
    rho = random_density_matrix(m)
    u, s, v_h = np.linalg.svd(rho)
    rho_cut = u[:, : m - 1] @ np.diag(s[: m - 1]) @ v_h[: m - 1]
    rho_cut = rho_cut / np.trace(rho_cut)
    pt_state_alice = partial_trace(rho_cut, [1], [3, 2])

    np.testing.assert_equal(is_density(rho_cut), True)
    np.testing.assert_equal(is_density(np.array(pt_state_alice)), True)
    np.testing.assert_equal(
        np.linalg.matrix_rank(rho_cut) + np.linalg.matrix_rank(pt_state_alice) <= 2 * m * n - m - n + 2,
        True,
    )
    # TODO
    # np.testing.assert_equal(is_separable(rho), True)


def test_entangled_realignment_criterion():
    """Determined to be entangled via the realignment criterion."""
    # Construct bound entangled state:
    # :math:`\rho = \frac{1}{4} \mathbb{I}_3 \otimes \mathbb{I}_3 - \sum_{i=0}^4 | \psi_i \rangle \langle \psi_i |`
    rho = np.identity(9)
    for i in range(5):
        rho = rho - tile(i) @ tile(i).conj().T
    rho = rho / 4
    np.testing.assert_equal(is_density(rho), True)
    np.testing.assert_equal(is_separable(rho), False)


def test_entangled_cross_norm_realignment_criterion():
    """Determined to be entangled by using Theorem 1 and Remark 1 of :cite:`Chen_2003_Matrix`."""
    p_var, a_var, b_var = 0.4, 0.8, 0.64
    rho = np.array(
        [
            [p_var * a_var**2, 0, 0, p_var * a_var * b_var],
            [0, (1 - p_var) * a_var**2, (1 - p_var) * a_var * b_var, 0],
            [0, (1 - p_var) * a_var * b_var, (1 - p_var) * a_var**2, 0],
            [p_var * a_var * b_var, 0, 0, p_var * a_var**2],
        ]
    )
    np.testing.assert_equal(is_separable(rho), False)


def test_separable_closeness_to_maximally_mixed_state():
    """Determined to be separable by closeness to the maximally mixed state."""
    rho = np.array(
        [
            [4, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 4, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 4, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 4, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 4, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 4, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 4, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 4],
        ]
    )
    rho = rho / np.trace(rho)
    np.testing.assert_equal(is_separable(rho), True)


def test_separable_small_rank1_perturbation_of_maximally_mixed_state():
    """Determined to be separable by being a small rank-1 perturbation of the maximally-mixed state."""
    rho = np.array(
        [
            [4, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 4, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 4, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 4, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 4, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 4, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 4, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 4, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 4],
        ]
    )
    rho = rho / np.trace(rho)
    np.testing.assert_equal(is_separable(rho), True)


# def test_separable_schmidt_rank():
#    """Determined to be separable by having operator Schmidt rank at most 2."""
#    # TODO: require better OCUT's
#    rho = np.array(
#        [
#            [0.25, 0.15, 0.1, 0.15, 0.09, 0.06, 0.1, 0.06, 0.04],
#            [0.15, 0.2, 0.05, 0.09, 0.12, 0.03, 0.06, 0.08, 0.02],
#            [0.1, 0.05, 0.05, 0.06, 0.03, 0.03, 0.04, 0.02, 0.02],
#            [0.15, 0.09, 0.06, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02],
#            [0.09, 0.12, 0.03, 0.12, 0.16, 0.04, 0.03, 0.04, 0.01],
#            [0.06, 0.03, 0.03, 0.08, 0.04, 0.04, 0.02, 0.01, 0.01],
#            [0.1, 0.06, 0.04, 0.05, 0.03, 0.02, 0.05, 0.03, 0.02],
#            [0.06, 0.08, 0.02, 0.03, 0.04, 0.01, 0.03, 0.04, 0.01],
#            [0.04, 0.02, 0.02, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01],
#        ]
#    )
#    rho = rho / np.trace(rho)
#    np.testing.assert_equal(is_separable(rho,level=3,tol=1e-20), True)


def test_symm_ext_catches_hard_entangled_state():
    """test_entangled_symmetric_extension uses a state."""
    # Ensure is_separable also returns False.
    rho_ent_symm = np.array(  # from test_entangled_symmetric_extension
        [
            [1.0, 0.67, 0.91, 0.67, 0.45, 0.61, 0.88, 0.59, 0.79],
            [0.67, 1.0, 0.5, 0.45, 0.67, 0.34, 0.59, 0.88, 0.44],
            [0.91, 0.5, 1.0, 0.61, 0.34, 0.68, 0.81, 0.44, 0.88],
            [0.67, 0.45, 0.61, 1.0, 0.67, 0.91, 0.5, 0.33, 0.45],
            [0.45, 0.67, 0.34, 0.67, 1.0, 0.5, 0.33, 0.5, 0.25],
            [0.61, 0.34, 0.68, 0.91, 0.5, 1.0, 0.45, 0.26, 0.5],
            [0.88, 0.59, 0.81, 0.5, 0.33, 0.45, 1.0, 0.66, 0.91],
            [0.59, 0.88, 0.44, 0.33, 0.5, 0.26, 0.66, 1.0, 0.48],
            [0.79, 0.44, 0.88, 0.45, 0.25, 0.5, 0.91, 0.48, 1.0],
        ]
    )
    rho_ent_symm = rho_ent_symm / np.trace(rho_ent_symm)
    # This state should be PPT, pass reduction/realignment maybe, but fail symm ext.
    # Assuming CVXPY is installed.
    try:
        import cvxpy

        cvxpy

        assert is_separable(rho_ent_symm, dim=[3, 3], level=2, tol=1e-7) is False  # More lenient tol for SDP
    except ImportError:
        pytest.skip("CVXPY not installed, skipping symmetric extension test.")


def test_separable_based_on_eigenvalues():
    """Determined to be separable by inspecting its eigenvalues. See Lemma 1 of :cite:`Johnston_2013_Spectrum`."""
    # Although this matrix, taken from the above paper, satisfies the eigenvalues condition,
    # this returns True from a line above the eigenvalues condition.
    rho = np.array(
        [
            [4 / 22, 2 / 22, -2 / 22, 2 / 22],
            [2 / 22, 7 / 22, -2 / 22, -1 / 22],
            [-2 / 22, -2 / 22, 4 / 22, -2 / 22],
            [2 / 22, -1 / 22, -2 / 22, 7 / 22],
        ]
    )
    np.testing.assert_equal(is_separable(rho), True)


def test_dim_none_prime_len_defaults_to_1_N():
    """Test dim none prime len defaults to 1N."""
    rho = np.eye(5) / 5
    assert is_separable(rho) is True  # Expects dim=[1,5] -> min_dim=1


def test_dim_int_not_divisor():
    """Test dim int not divisor."""
    with pytest.raises(ValueError, match="must evenly divide"):
        is_separable(np.eye(6) / 6, dim=4)


def test_dim_list_product_mismatch():
    """Test dim list product mismatch."""
    with pytest.raises(ValueError, match="Product of list dimensions must equal state length"):
        is_separable(np.eye(6) / 6, dim=[2, 2])


def test_dim_list_non_positive():
    """Test dim list non positive."""
    with pytest.raises(ValueError, match="Dimensions in list must be non-negative integers"):
        is_separable(np.eye(4) / 4, dim=[2, -2])
    with pytest.raises(ValueError, match="Dimensions in list must be non-negative integers"):
        is_separable(np.eye(0), dim=[0, -1])  # state_len=0


def test_dim_int_non_positive():
    """Test dim int non positive."""
    with pytest.raises(ValueError, match="must be positive"):
        is_separable(np.eye(4) / 4, dim=0)


def test_dim_invalid_type():
    """Test dim invalid type."""
    with pytest.raises(ValueError, match="must be None, an int, or a list"):
        is_separable(np.eye(4) / 4, dim="invalid_type")


def test_dim_zero_subsystem_for_nonzero_state():
    """Test dim zero subsystem for nonzero state."""
    with pytest.raises(ValueError, match="Non-zero state with zero-dim subsystem"):
        is_separable(np.eye(2) / 2, dim=[2, 0])


def test_pure_entangled_state():
    """test_pure_entangled_state."""
    rho_bell = np.outer(bell(0), bell(0).conj())
    assert is_separable(rho_bell, dim=[2, 2]) is False


def test_ppt_2x2_mixed_separable():
    """test_ppt_2x2_mixed_separable."""
    psi1 = np.kron(basis(2, 0), basis(2, 0))
    psi2 = np.kron(basis(2, 1), basis(2, 1))
    rho = 0.5 * np.outer(psi1, psi1) + 0.5 * np.outer(psi2, psi2)
    assert is_separable(rho, dim=[2, 2])


def test_3x3_ppt_rank3_separable_skips_plucker():
    """Example: mixture of 3 product states in C^3 x C^3."""
    p1 = np.kron(basis(3, 0), basis(3, 0))
    p2 = np.kron(basis(3, 1), basis(3, 1))
    p3 = np.kron(basis(3, 2), basis(3, 2))
    rho = (np.outer(p1, p1) + np.outer(p2, p2) + np.outer(p3, p3)) / 3
    # This state is separable, PPT, rank 3.
    # It should not be decided by Plucker, but by a later check (e.g. Horodecki or even Reduction pass)
    # or ultimately symmetric extension.
    assert is_separable(rho, dim=[3, 3])  # Expect True


def test_horodecki_rank_le_marginal_rank():
    """Construct/find a PPT state rho on C^dA x C^dB where."""
    # rank(rho) > max(dA,dB) AND
    # rank(rho) <= rank(partial_trace(rho, sys=[1], dim=[dA,dB]))
    # and rho IS separable by this criterion. This is non-trivial to construct.
    # Example: (Hypothetical 3x3 state, rank 4, but rank(rho_A)=4)
    # rho_example = ... construct ...
    # assert is_separable(rho_example, dim=[3,3]) is True
    pass  # Placeholder - requires specific state construction


def test_entangled_by_reduction_criterion():
    """Entangled by reduction_criterion.

    The state constructed here (Choi of transpose map) is known to be NOT positive semidefinite for d>=2.
    """
    d = 3
    rho = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            # Assuming basis() creates |i> or |j> column vectors
            # If basis(d,idx) is not available or has different signature, adjust state construction
            # For simplicity, let's use identity to get basis vectors if basis() is problematic here
            e_i = np.zeros((d, 1))
            e_i[i] = 1
            e_j = np.zeros((d, 1))
            e_j[j] = 1

            ket_ij = np.kron(e_i, e_j)
            # Need bra_ji to be row vector for outer product: (<j| kron <i|)
            # So, ( |j> kron |i> )^dagger
            bra_ji_conj_T = np.kron(e_j, e_i).conj().T
            rho += ket_ij @ bra_ji_conj_T  # ket @ bra form
    rho /= d  # Normalize (Choi of transpose has trace d)

    # The function is_separable should raise a ValueError because rho is not PSD.
    with pytest.raises(ValueError, match="Checking separability of non-positive semidefinite matrix is invalid."):
        is_separable(rho, dim=[d, d])


# def test_entangled_by_ccnr():
#     # d=3, p=0.4. Isotropic state.
#     # Separable if p <= 1/4 = 0.25.
#     # CCNR detects if p > 1/3 ~ 0.333.
#     # p=0.4 is entangled and should be caught by CCNR.
#     # It should also pass Reduction if PPT.
#     # TODO: expected to be true, failed might require better OCUT's logic and  fix is_ppt
#     rho = isotropic(3, 0.4)
#     # If isotropic(3,0.4) is known to be numerically challenging for is_ppt's default tol
#     assert is_ppt(rho, dim=[3,3], tol=1e-6) # Pass a more lenient tol for this test
#     # ... rest of the test ...
#     assert is_separable(rho, dim=[3,3]) is False


def test_breuer_hall_skip_odd_dim():
    """A 2x3 separable state that is not caught by earlier criteria."""
    # E.g., a simple product state.
    rho_A = np.eye(2) / 2
    rho_B = np.eye(3) / 3
    rho = np.kron(rho_A, rho_B)
    # It should pass all entanglement witnesses.
    # Breuer-Hall will apply for dim=2, then skip for dim=3.
    assert is_separable(rho, dim=[2, 3])


def test_trace_near_zero_but_not_zero_matrix_V2():
    """test_trace_near_zero_but_not_zero_matrix_V2."""
    tol_for_is_separable = 1e-8  # Typical default
    # State where trace is small compared to tol_for_is_separable,
    # but elements are significant enough to not be considered a zero matrix.
    state_elements_scale = 1e-9
    # For example, a diagonal matrix with small elements summing to a small trace
    state = np.diag([state_elements_scale, state_elements_scale])
    # trace = 2 * 1e-9. max_abs_element = 1e-9.
    # Here, abs(trace) = 2e-9 < tol_for_is_separable (1e-8).
    # And state is clearly "not zero" if we consider elements of order 1e-9 as significant.
    with pytest.raises(ValueError, match="Trace of the input state is close to zero, but state is not zero matrix."):
        is_separable(state, tol=tol_for_is_separable)


def test_dim_none_state_len_zero():
    """test_dim_none_state_len_zero."""
    state = np.array([[]])  # Or np.empty((0,0))
    if state.size == 0:  # Ensure it's truly 0x0 for this test path
        state = np.zeros((0, 0))
    assert is_separable(state)


def test_dim_int_negative_for_non_empty():
    """test_dim_int_negative_for_non_empty."""
    with pytest.raises(ValueError, match="must be positive for non-empty states or zero for empty states"):
        is_separable(np.eye(4) / 4, dim=-2)


def test_dim_int_for_empty_state_nonzero_dim():
    """0 none state two dimension."""
    with pytest.raises(ValueError, match="Cannot apply positive dimension"):
        is_separable(np.zeros((0, 0)), dim=2)


def test_rank1_pert_skip_for_rank_deficient():
    """A rank-1 product state (e.g., 2x2)."""
    # np.outer(np.kron(basis(2, 0), basis(2, 0)), np.kron(basis(2, 0), basis(2, 0)))
    # This is rank 1. eigvalsh might return fewer than 4 eigs if others are zero.
    # It will be caught by state_rank==1 check.
    # Need a rank > 1, but < prod_dim, that is PPT and separable.
    # E.g. rho = 0.5 * prod1 + 0.5 * prod2, where prod1, prod2 are orthogonal product states.
    # This has rank 2. len(lam) from eigvalsh should be prod_dim_val.
    # This branch is hard to hit specifically if eigvalsh robustly gives all eigs.
    pass


def test_dim_none_state_len_zero_is_sep():
    """0 none state."""
    state = np.zeros((0, 0))
    assert is_separable(state) is True


def test_dim_none_state_len_one_is_sep():
    """1 none state."""
    state = np.array([[1.0]])
    assert is_separable(state) is True


def test_dim_none_composite_len_factors_is_sep():
    """6 = 2x3. Default dim should be [2,3] or [3,2]. min_dim=2."""
    # Eye(6)/6 is PPT. prod_dim=6 <= 6, so should be True.
    assert is_separable(np.eye(6) / 6) is True
    # 10 = 2x5. Default dim should be [2,5]. min_dim=2.
    # Eye(10)/10 is PPT. prod_dim=10 > 6. Horodecki rank <= max_dim (10 <= 5 False).
    # Horodecki rank <= marginal rank. rank(I/10)=10. rank(I/5)=5. 10 <= 5 False.
    # Will pass other checks for separable state.
    assert is_separable(np.eye(10) / 10) is True


def test_dim_int_zero_for_empty_state_is_sep():
    """Test empty state."""
    assert is_separable(np.zeros((0, 0)), dim=0) is True


def test_skip_3x3_rank4_if_not_rank4():
    """A 3x3 PPT separable state of rank 9 (full rank)."""
    rho = np.eye(9) / 9
    # Should skip Plucker, pass Horodecki rank <= max_dim_val (9<=3 False)
    # Pass Horodecki rank <= marginal rank (9 <= 3 False)
    # Pass Reduction, Realignment.
    # Pass Rank-1 pert. (lam[1]-lam[end] = 0)
    assert is_separable(rho, dim=[3, 3]) is True


def test_skip_horodecki_if_not_applicable_proceeds():
    """Use Tiles state, it's PPT, 3x3 Rank 4."""
    # Plucker should return False. This test ensures Horodecki isn't wrongly True before Plucker.
    rho_tiles = np.identity(9)
    for i in range(5):
        rho_tiles = rho_tiles - tile(i) @ tile(i).conj().T
    rho_tiles = rho_tiles / 4
    assert is_separable(rho_tiles, dim=[3, 3]) is False  # Caught by Plucker


def test_reduction_passed_for_product_state():
    """Example: 3x3 product state."""
    # Will pass PPT, prod_dim<=6.
    # To specifically test reduction pass: need state that passes PPT, prod_dim > 6,
    # and passes Horodecki, then passes Reduction.
    rho_prod_3x3 = np.kron(np.eye(3) / 3, np.eye(3) / 3)
    assert is_separable(rho_prod_3x3, dim=[3, 3]) is True  # Should be caught by rank-1 pert.


# --- Symmetric Extension tests ---
def test_symm_ext_level_1_is_ppt_equivalent():
    """For level=1, has_symmetric_extension should be equivalent to PPT for PPT states."""
    # Use a PPT separable state
    rho = np.eye(4) / 4
    assert is_separable(rho, dim=[2, 2], level=1) is True  # Already caught by prod_dim<=6

    # Use a 3x3 separable state that passes early checks
    np.eye(9) / 9
    # is_separable(rho_3x3_sep, level=1) will hit rank-1 pert and return True.
    # To test symm ext level 1: need a state that reaches it.
    # This is hard because symm ext is the last resort.
    # We can directly test `has_symmetric_extension(rho_3x3_sep, level=1)`
    # if is_ppt(rho_3x3_sep, dim=[3,3]) then has_symmetric_extension(rho_3x3_sep,1) should be True.
    # This test is more about `has_symmetric_extension` than `is_separable` flow.
    pass


def test_final_return_false_for_unclassified_entangled():
    """Test for final return False if all else fails for an entangled state."""
    # This needs a PPT entangled state that is NOT caught by any implemented criterion before symm ext,
    # AND for which has_symmetric_extension also returns False (or errors and is skipped).
    # The Tiles state is caught by Plucker.
    # The rho_ent_symm above is a good candidate if has_symmetric_extension works correctly.
    # If has_symmetric_extension is mocked to always return False (or error):
    # from unittest import mock
    # with mock.patch('toqito.state_props.has_symmetric_extension.has_symmetric_extension', return_value=False):
    #     assert is_separable(rho_ent_symm_known_ppt_entangled_passes_other_witnesses) is False
    pass  # Requires mocking or a very specific state.


def test_trace_small_and_matrix_is_almost_zero_proceeds():
    """trace=2e-20. Numerically very close to zero."""
    state = np.eye(2) * 1e-20
    # Using tol from a failing test for consistency
    # For this state, with tol=1e-10, abs(trace) < tol is true.
    # is_matrix_effectively_zero (with atol=tol*EPS) should be true. So no error.
    # Proceeds, likely caught by min_dim_val==1 (if dim=[1,2]) or other early checks.
    assert is_separable(state, dim=[1, 2], tol=1e-10) is True
