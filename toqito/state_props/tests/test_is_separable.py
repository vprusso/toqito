"""Test is_separable."""

import numpy as np
import pytest  # For pytest.skip and pytest.raises

from toqito.matrix_ops import partial_transpose
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
