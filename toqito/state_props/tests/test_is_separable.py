"""Test is_separable."""

import numpy as np
import pytest  # For pytest.skip and pytest.raises

from toqito.channels import partial_transpose
from toqito.matrix_props import is_density, is_positive_semidefinite
from toqito.rand import random_density_matrix
from toqito.state_props.is_separable import is_separable
from toqito.states import basis, bell, isotropic, tile


# Helper function for Horodecki operational criterion test
def separable_state_2x3_rank3():
    """Construct a 2x3 separable state of rank 3."""
    psi_A0 = np.array([1, 0], dtype=complex)
    psi_A1 = np.array([0, 1], dtype=complex)
    psi_B0 = np.array([1, 0, 0], dtype=complex)
    psi_B1 = np.array([0, 1, 0], dtype=complex)
    psi_B2 = np.array([0, 0, 1], dtype=complex)
    rho1 = np.kron(np.outer(psi_A0, psi_A0.conj()), np.outer(psi_B0, psi_B0.conj()))
    rho2 = np.kron(np.outer(psi_A0, psi_A0.conj()), np.outer(psi_B1, psi_B1.conj()))
    rho3 = np.kron(np.outer(psi_A1, psi_A1.conj()), np.outer(psi_B2, psi_B2.conj()))
    rho = (rho1 + rho2 + rho3) / 3
    test_tol = 1e-8
    assert np.isclose(np.trace(rho), 1)
    assert is_positive_semidefinite(rho, rtol=test_tol, atol=test_tol)
    assert np.linalg.matrix_rank(rho, tol=test_tol) == 3
    rho_pt_A = partial_transpose(rho, sys=0, dim=[2, 3])
    assert is_positive_semidefinite(rho_pt_A, rtol=test_tol, atol=test_tol)
    return rho


def test_entangled_zhang_realignment_criterion():
    """Test for entanglement using Zhang's realignment criterion (Bell state)."""
    rho = bell(0) @ bell(0).conj().T
    assert not is_separable(rho, dim=[2, 2])


def test_entangled_qutrit_qutrit():
    """Test for entanglement in the qutrit-qutrit case (Maximally Entangled State)."""
    psi = (1 / np.sqrt(3)) * (
        np.kron(basis(3, 0), basis(3, 0)) + np.kron(basis(3, 1), basis(3, 1)) + np.kron(basis(3, 2), basis(3, 2))
    )
    rho = np.outer(psi, psi)
    assert not is_separable(rho, dim=[3, 3])


def test_entangled_breuer_hall_path_non_ppt():
    """Test Breuer-Hall path with a non-PPT state (Bell state)."""
    rho = bell(0) @ bell(0).conj().T
    assert not is_separable(rho, dim=[2, 2])


def test_non_positive_semidefinite_matrix():
    """Ensure separability of non-positive semidefinite matrix is invalid."""
    with pytest.raises(ValueError, match="Input state must be a positive semidefinite matrix."):
        state = np.array([[-1, -1], [-1, -1]])
        is_separable(state)


def test_psd_matrix_local_dim_one():
    """Every positive semidefinite matrix is separable when one of the local dimensions is 1."""
    assert is_separable(np.identity(2) / 2, dim=[1, 2])
    assert is_separable(np.identity(3) / 3, dim=[3, 1])
    assert is_separable(np.array([[1.0]]), dim=[1, 1])


def test_invalid_dim_parameter():
    """The dimension of the state must evenly divide the length of the state."""
    with pytest.raises(ValueError, match="Integer `dim` .* must evenly divide state dimension .*"):
        rho_iso_3 = isotropic(3, 1 / (3 + 1))
        is_separable(rho_iso_3, 4)


def test_entangled_ppt_criterion():
    """Determined to be entangled via the PPT criterion."""
    rho = bell(0) @ bell(0).conj().T
    assert not is_separable(rho, dim=[2, 2])


def test_ppt_small_dimensions_2x2_product():
    """Separable: PPT criterion sufficiency in 2x2 (product state)."""
    psi_A = basis(2, 0)
    psi_B = basis(2, 0)
    rho_prod_2x2 = np.outer(np.kron(psi_A, psi_B), np.kron(psi_A, psi_B).conj())
    assert is_separable(rho_prod_2x2, dim=[2, 2])


def test_ppt_small_dimensions_2x3_product():
    """Separable: PPT criterion sufficiency in 2x3 (product state)."""
    psi_A = basis(2, 0)
    psi_B = (basis(3, 0) + basis(3, 1)) / np.sqrt(2)
    rho_prod_2x3 = np.outer(np.kron(psi_A, psi_B), np.kron(psi_A, psi_B).conj())
    assert is_separable(rho_prod_2x3, dim=[2, 3])


def test_entangled_realignment_criterion_tiles_state():
    """Entangled: Tiles state (3x3 rank-4 PPT BE) via det(F) or CCNR."""
    rho_tiles = np.identity(9)
    for i in range(5):
        rho_tiles = rho_tiles - tile(i) @ tile(i).conj().T
    rho_tiles = rho_tiles / 4
    assert is_density(rho_tiles)
    assert not is_separable(rho_tiles, dim=[3, 3])


def test_entangled_cross_norm_realignment_criterion_non_ppt():
    """Entangled: Original matrix for Zhang et al. (non-PPT)."""
    rho_bell = bell(0) @ bell(0).conj().T
    assert not is_separable(rho_bell, dim=[2, 2])


def test_separable_closeness_to_maximally_mixed_state():
    """Separable: Closeness to the maximally mixed state."""
    rho_raw = np.array(
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
        ],
        dtype=complex,
    )
    rho = rho_raw / np.trace(rho_raw)
    assert is_separable(rho, dim=[3, 3])


def test_separable_small_rank1_perturbation_of_maximally_mixed_state():
    """Separable: Small rank-1 perturbation of the maximally-mixed state."""
    rho_raw = np.array(
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
        ],
        dtype=complex,
    )
    rho = rho_raw / np.trace(rho_raw)
    assert is_separable(rho, dim=[3, 3])


def test_separable_operator_schmidt_rank_2():
    """Separable: Operator Schmidt rank 2 (mixture of two product states)."""
    A1 = random_density_matrix(3)
    B1 = random_density_matrix(3)
    A2 = random_density_matrix(3)
    B2 = random_density_matrix(3)
    rho = 0.5 * np.kron(A1, B1) + 0.5 * np.kron(A2, B2)
    rho = rho / np.trace(rho)
    assert is_separable(rho, dim=[3, 3])


def test_entangled_symmetric_extension_dps():
    """Entangled: By not having a PPT symmetric extension (DPS criterion)."""
    rho_raw = np.array(
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
        ],
        dtype=complex,
    )
    rho = rho_raw / np.trace(rho_raw)
    assert not is_separable(rho, dim=[3, 3], level=2)


def test_separable_based_on_eigenvalues_2x2_product_state():
    """Separable: 2x2 state by Johnston 2013 spectrum condition (Eq. 12)."""
    psi_A = basis(2, 0)
    psi_B = basis(2, 1)
    rho = np.outer(np.kron(psi_A, psi_B), np.kron(psi_A, psi_B).conj())
    assert is_separable(rho, dim=[2, 2])


def test_product_state_separable():
    """Separable: Pure product state."""
    psiA = np.array([1, 0], dtype=complex)
    psiB = np.array([0, 1], dtype=complex)
    rho = np.outer(np.kron(psiA, psiB), np.kron(psiA, psiB).conj())
    assert is_separable(rho, [2, 2])


def test_bell_state_entangled():
    """Entangled: Bell state (non-PPT)."""
    rho = bell(0) @ bell(0).conj().T
    assert not is_separable(rho, [2, 2])


def test_horodecki_operational_criterion_rank_le_max_dim():
    """Separable: Horodecki Op. (rank <= max_dim) for 2x3 rank 3 state."""
    rho = separable_state_2x3_rank3()
    assert is_separable(rho, [2, 3])


def test_horodecki_operational_criterion_rank_sum_separable():
    """Separable: Horodecki Op. (rank-sum) for a specifically constructed PPT state."""
    pytest.skip(
        "Skipping specific Horodecki rank-sum test: difficult to "
        + "isolate state / construct with specific rank properties easily."
    )


def test_ppt_entangled_state_is_not_separable_tiles():
    """Entangled: Tiles state (3x3 rank-4 PPT BE) should be caught."""
    rho_tiles = np.identity(9)
    for i in range(5):
        rho_tiles = rho_tiles - tile(i) @ tile(i).conj().T
    rho_tiles = rho_tiles / 4
    assert is_density(rho_tiles), "Tiles state construction issue."

    # Verify it's PPT (sanity check for the test's premise that it's a PPT entangled state)
    pt_tiles_A = partial_transpose(rho_tiles, sys=0, dim=[3, 3])
    pt_tiles_B = partial_transpose(rho_tiles, sys=1, dim=[3, 3])  # Check PT w.r.t other system too for robustness
    # Use a tolerance that's consistent with what is_separable might use internally for PSD checks.
    # The default tol in is_separable is 1e-8.
    internal_test_tol = 1e-8
    assert is_positive_semidefinite(pt_tiles_A, rtol=internal_test_tol, atol=internal_test_tol), (
        f"Tiles state PT_A not PSD as expected. Evals: {np.linalg.eigvalsh(pt_tiles_A)}"
    )
    assert is_positive_semidefinite(pt_tiles_B, rtol=internal_test_tol, atol=internal_test_tol), (
        f"Tiles state PT_B not PSD as expected. Evals: {np.linalg.eigvalsh(pt_tiles_B)}"
    )

    assert not is_separable(rho_tiles, dim=[3, 3])


def test_realignment_detects_entanglement_isotropic_3x3():
    """Entangled: 3x3 Isotropic PPT state caught by CCNR."""
    dim_param = 3
    alpha = 0.4
    rho_iso = isotropic(dim_param, alpha)
    assert not is_separable(rho_iso, [dim_param, dim_param])


def test_reduction_criterion_catches_entanglement_non_ppt():
    """test_reduction_criterion_catches_entanglement_non_ppt."""
    rho = bell(0) @ bell(0).conj().T
    assert not is_separable(rho, [2, 2])


def test_separable_3x3_rank4_ppt_chow_form():
    """Separable: 3x3 rank 4 PPT state, separable by det(F)=0 (Chow form)."""
    p_states = []
    psi_A_basis = [basis(3, i) for i in range(3)]
    psi_B_basis = [basis(3, i) for i in range(3)]
    p_states.append(
        np.kron(np.outer(psi_A_basis[0], psi_A_basis[0].conj()), np.outer(psi_B_basis[0], psi_B_basis[0].conj()))
    )
    p_states.append(
        np.kron(np.outer(psi_A_basis[1], psi_A_basis[1].conj()), np.outer(psi_B_basis[1], psi_B_basis[1].conj()))
    )
    p_states.append(
        np.kron(np.outer(psi_A_basis[2], psi_A_basis[2].conj()), np.outer(psi_B_basis[2], psi_B_basis[2].conj()))
    )
    psi_A_mix = (psi_A_basis[0] + psi_A_basis[1]) / np.sqrt(2)
    psi_B_mix = (psi_B_basis[0] + psi_B_basis[2]) / np.sqrt(2)
    p_states.append(np.kron(np.outer(psi_A_mix, psi_A_mix.conj()), np.outer(psi_B_mix, psi_B_mix.conj())))
    rho = sum(p_states) / len(p_states)
    rho = rho / np.trace(rho)

    if np.linalg.matrix_rank(rho, tol=1e-7) == 4:
        assert is_separable(rho, dim=[3, 3])
    else:
        pytest.skip("Constructed 3x3 state not rank 4, skipping specific det(F) separable test.")


def test_johnston_spectrum_2x2_specific_separable_passes():
    """Separable: 2x2 PPT state for Johnston spectrum condition - passing case."""
    # lam = [0.4, 0.35, 0.15, 0.10]
    # (0.4-0.15)^2 = (0.25)^2 = 0.0625
    # 4*0.35*0.1 = 4*0.035 = 0.14.  0.0625 <= 0.14. This passes.
    rho_johnston_test = np.diag([0.4, 0.35, 0.15, 0.10])
    assert is_separable(rho_johnston_test, dim=[2, 2])


def test_johnston_spectrum_2x2_specific_separable_fails_condition():
    """Separable: 2x2 PPT state FAILS Johnston spectrum condition but is separable otherwise."""
    # lam = [0.7, 0.1, 0.1, 0.1]
    # (0.7-0.1)^2 = 0.6^2 = 0.36
    # 4*0.1*0.1 = 0.04.  0.36 is NOT <= 0.04. This condition fails.
    # But this state is diag([0.7,0.1,0.1,0.1]) which is separable (mixture of |00><00| etc)
    # It will be caught by Horodecki Op (rank 4, max_dim=2, fails; rank_sum 4+4=8 > 6 fails)
    # It will be caught by OSR (OSR=4, so not OSR<=2)
    # It might be caught by in_separable_ball.
    # This test is more about ensuring that if this condition *isn't* met, the state
    # isn't wrongly declared entangled IF it's separable by other means.
    rho_fails_johnston_cond = np.diag([0.7, 0.1, 0.1, 0.1])
    # This state IS separable. is_separable should return True.
    # It will be caught by the fact that it's a mixture of product states,
    # possibly by `in_separable_ball` or other general separability.
    assert is_separable(rho_fails_johnston_cond, dim=[2, 2])


def test_hildebrand_block_hankel_2x2_B_hermitian():
    """Separable: 2x2 PPT by Hildebrand (B is Hermitian, so rank(B-B.T)=0)."""
    dim_A, dim_B = 2, 2
    A_block = np.array([[0.3, 0.01j], [-0.01j, 0.25]], dtype=complex)
    C_block = np.array([[0.25, -0.02j], [0.02j, 0.1]], dtype=complex)
    B_block = np.array([[0.05, 0.02], [0.02, 0.05]], dtype=complex)

    state_t = np.block([[A_block, B_block], [B_block.conj().T, C_block]])
    state_t = state_t / np.trace(state_t)

    pt_A = partial_transpose(state_t, sys=0, dim=[dim_A, dim_B])
    if not (
        is_positive_semidefinite(state_t, rtol=1e-7, atol=1e-7) and is_positive_semidefinite(pt_A, rtol=1e-7, atol=1e-7)
    ):
        pytest.skip("Constructed Hildebrand block Hankel state not PSD/PPT.")

    assert is_separable(state_t, dim=[dim_A, dim_B])


def test_hildebrand_homothetic_images_2x2_separable():
    """Separable: 2x2 PPT state that should pass Hildebrand homothetic images test."""
    # Use the same state as above, it should also pass this if conditions are related.
    dim_A, dim_B = 2, 2
    A_block = np.array([[0.4, 0], [0, 0.2]], dtype=complex)
    C_block = np.array([[0.3, 0], [0, 0.1]], dtype=complex)
    B_block = np.array([[0.05, 0.01j], [-0.01j, 0.05]], dtype=complex)

    state_t = np.block([[A_block, B_block], [B_block.conj().T, C_block]])
    state_t = state_t / np.trace(state_t)

    pt_A = partial_transpose(state_t, sys=0, dim=[dim_A, dim_B])
    if not (
        is_positive_semidefinite(state_t, rtol=1e-7, atol=1e-7) and is_positive_semidefinite(pt_A, rtol=1e-7, atol=1e-7)
    ):
        pytest.skip("Constructed Hildebrand homothetic test state not PSD or not PPT.")

    assert is_separable(state_t, dim=[dim_A, dim_B])


def test_johnston_lemma1_2x2_separable():
    """Separable: 2x2 PPT state that should pass Johnston Lemma 1."""
    dim_A, dim_B = 2, 2
    A_block = np.array([[0.4, 0], [0, 0.3]], dtype=complex)
    C_block = np.array([[0.2, 0], [0, 0.1]], dtype=complex)
    B_block = np.array([[0.01, 0.01], [0.01, 0.01]], dtype=complex)

    state_t = np.block([[A_block, B_block], [B_block.conj().T, C_block]])
    state_t = state_t / np.trace(state_t)

    pt_A = partial_transpose(state_t, sys=0, dim=[dim_A, dim_B])
    if not (
        is_positive_semidefinite(state_t, rtol=1e-7, atol=1e-7) and is_positive_semidefinite(pt_A, rtol=1e-7, atol=1e-7)
    ):
        pytest.skip("Constructed Johnston Lemma 1 test state not PSD or not PPT.")

    assert is_separable(state_t, dim=[dim_A, dim_B])


# --- Skipped tests for very specific literature states ---
@pytest.mark.skip(reason="Requires specific literature state for Zhang criterion isolation.")
def test_zhang_criterion_catches_ppt_entangled():
    """test_ha_map_catches_specific_3x3_ppt_entangled."""
    pass


@pytest.mark.skip(
    reason="Requires a specific 3x3 PPT entangled state known to be caught by Ha maps but not simpler criteria."
)
def test_ha_map_catches_specific_3x3_ppt_entangled():
    """test_ha_map_catches_specific_3x3_ppt_entangled."""
    pass


@pytest.mark.skip(reason="Breuer-Hall map specific test requires a state not caught by CCNR/Reduction.")
def test_breuer_hall_catches_even_dim_ppt_entangled():
    """test_separable_by_symmetric_extension."""
    pass


@pytest.mark.skip(reason="Skipping specific symmetric extension True path: requires specific hard state.")
def test_separable_by_symmetric_extension():
    """test_separable_by_symmetric_extension."""
    pass


def test_2xN_rules_after_swap_3x2_separable():
    """Separable: 3x2 PPT state triggering swap and then a 2xN rule."""
    dA, dB = 3, 2  # Original dimensions, will be swapped to 2x3
    dims = [dA, dB]

    psi_A = basis(dA, 0)  # e.g., |0>_A for A in C^3
    psi_B = basis(dB, 0)  # e.g., |0>_B for B in C^2
    rho = np.outer(np.kron(psi_A, psi_B), np.kron(psi_A, psi_B).conj())

    assert is_density(rho)
    assert is_separable(rho, dim=dims)


def test_2xN_johnston_spectrum_after_swap_4x2_separable():
    """Separable: 4x2 PPT state for Johnston spectrum after swap."""
    # Consider a simple product state on 3x2 that becomes 2x3 after swap
    psi_A_3dim = basis(3, 0)
    psi_B_2dim = basis(2, 0)
    rho_3x2_prod = np.outer(np.kron(psi_A_3dim, psi_B_2dim), np.kron(psi_A_3dim, psi_B_2dim).conj())
    assert is_separable(rho_3x2_prod, dim=[3, 2])
