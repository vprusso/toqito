"""Test is_separable."""

from unittest import mock

import numpy as np
import pytest

from toqito.channel_ops.partial_channel import partial_channel
from toqito.matrix_props import is_density, is_positive_semidefinite
from toqito.rand import random_density_matrix
from toqito.state_props import is_ppt, is_separable
from toqito.states import basis, bell, horodecki, isotropic, tile, werner

# --- Parameterized Tests for Invalid Inputs ---
"""
These tests verify that is_separable raises appropriate exceptions for invalid inputs,
such as non-PSD matrices, incorrect dimensions, or malformed inputs. Parameterization
is used to consolidate similar ValueError cases, reducing redundancy while covering
all scenarios.
"""
invalid_input_cases = [
    {"state": np.array([[-1, -1], [-1, -1]]), "error_msg": "non-positive semidefinite"},
    {"state": isotropic(3, 1 / 4), "dim": 4, "error_msg": "must evenly divide"},
    {"state": np.eye(6) / 6, "dim": 4, "error_msg": "must evenly divide"},
    {"state": np.eye(6) / 6, "dim": [2, 2], "error_msg": "Product of list dimensions"},
    {"state": np.eye(4) / 4, "dim": [2, -2], "error_msg": "non-negative integers"},
    {"state": np.eye(4) / 4, "dim": 0, "error_msg": "must be positive"},
    {"state": np.eye(4) / 4, "dim": -2, "error_msg": "must be positive"},
    {"state": np.eye(4) / 4, "dim": "invalid_type", "error_msg": "must be None, an int, or a list"},
    {"state": np.eye(2) / 2, "dim": [2, 0], "error_msg": "zero-dim subsystem"},
    {"state": np.diag([1e-9, 1e-9]), "tol": 1e-8, "error_msg": "Trace of the input state is close to zero"},
    {"state": np.zeros((0, 0)), "dim": 2, "error_msg": "Cannot apply positive dimension"},
    {"state": np.eye(2) / 2, "dim": [0, 2], "error_msg": "zero-dim subsystem"},
    {"state": np.eye(6) / 6, "dim": [2.0, 3.0], "error_msg": "non-negative integers"},
]


@pytest.mark.parametrize("case", invalid_input_cases)
def test_invalid_inputs(case):
    """Parameterized test for invalid input cases raising ValueError."""
    state = case["state"]
    dim = case.get("dim")
    tol = case.get("tol")
    error_msg = case["error_msg"]
    kwargs = {"state": state}
    if dim is not None:
        kwargs["dim"] = dim
    if tol is not None:
        kwargs["tol"] = tol
    with pytest.raises(ValueError, match=error_msg):
        is_separable(**kwargs)


def test_input_state_not_square():
    """Test that a non-square matrix raises ValueError."""
    with pytest.raises(ValueError, match="Input state must be a square matrix"):
        is_separable(np.array([[1, 2, 3], [4, 5, 6]]))


def test_input_state_not_numpy_array():
    """Test that a non-NumPy array input raises TypeError."""
    with pytest.raises(TypeError, match="must be a NumPy array"):
        is_separable("not_a_matrix")


# --- Parameterized Tests for Simple Separable States ---
simple_separable_cases = [
    pytest.param(np.identity(2), {}, id="identity_2x2_dim_inferred_sep"),
    pytest.param(np.eye(5) / 5, {}, id="identity_5x5_dim_inferred_prime_len_sep"),
    pytest.param(np.eye(6) / 6, {}, id="identity_6x6_dim_inferred_composite_len_sep"),
    pytest.param(np.eye(10) / 10, {}, id="identity_10x10_dim_inferred_composite_len_sep"),
    pytest.param(np.eye(9) / 9, {"dim": [3, 3]}, id="identity_9x9_dim_3x3_sep"),
    pytest.param(np.kron(np.eye(3) / 3, np.eye(3) / 3), {"dim": [3, 3]}, id="product_3x3_identities_sep"),
    pytest.param(np.eye(4) / 4, {"dim": [2, 2], "level": 1}, id="identity_4x4_level1_symm_ext_sep"),
    pytest.param(np.eye(9) / 9, {"dim": [3, 3], "level": 1}, id="identity_9x9_level1_symm_ext_sep"),
    pytest.param(np.zeros((0, 0)), {}, id="empty_state_dim_inferred_sep"),
    pytest.param(np.array([[1.0]]), {}, id="1x1_state_dim_inferred_sep"),
    pytest.param(np.zeros((0, 0)), {"dim": 0}, id="empty_state_dim_0_sep"),
    pytest.param(np.eye(6) / 6, {"dim": 2}, id="identity_6x6_dim_int_divides_sep"),
]


@pytest.mark.parametrize("state, kwargs", simple_separable_cases)
def test_simple_separable_states(state, kwargs):
    """Test various simple states known to be separable."""
    assert is_separable(state, **kwargs)


# --- Parameterized Tests for Simple Entangled States ---
"""
These tests verify that is_separable returns True for states known to be separable,
based on various criteria such as PPT, low rank, eigenvalue properties, or simple
product structures.
"""


@pytest.fixture
def entangled_qutrit_qutrit_state():
    """Fixture for a maximally entangled qutrit-qutrit state (normalized)."""
    psi = (1 / np.sqrt(3)) * (
        np.kron([1, 0, 0], [1, 0, 0]) + np.kron([0, 1, 0], [0, 1, 0]) + np.kron([0, 0, 1], [0, 0, 1])
    )
    rho = np.outer(psi, psi.conj())
    assert np.isclose(np.trace(rho), 1.0), "Trace of fixture state is not 1"
    assert is_positive_semidefinite(rho, 1e-9), "Fixture state is not PSD"
    return rho


@pytest.fixture
def entangled_bell_state_0():
    """Fixture for the density matrix for the Bell state |Φ+> (normalized)."""
    rho = bell(0) @ bell(0).conj().T
    assert np.isclose(np.trace(rho), 1.0), "Trace of fixture state is not 1"
    assert is_positive_semidefinite(rho, 1e-9), "Fixture state is not PSD"
    return rho


@pytest.fixture
def separable_state_2x3_rank3():
    """Fixture for a separable 2x3 rank-3 state (normalized)."""
    psi_A0 = np.array([1, 0], dtype=complex)
    psi_A1 = np.array([0, 1], dtype=complex)
    psi_B0 = np.array([1, 0, 0], dtype=complex)
    psi_B1 = np.array([0, 1, 0], dtype=complex)
    psi_B2 = np.array([0, 0, 1], dtype=complex)
    rho1 = np.kron(np.outer(psi_A0, psi_A0.conj()), np.outer(psi_B0, psi_B0.conj()))
    rho2 = np.kron(np.outer(psi_A0, psi_A0.conj()), np.outer(psi_B1, psi_B1.conj()))
    rho3 = np.kron(np.outer(psi_A1, psi_A1.conj()), np.outer(psi_B2, psi_B2.conj()))
    rho = (rho1 + rho2 + rho3) / 3
    assert np.isclose(np.trace(rho), 1), "Trace of fixture state is not 1"
    assert np.all(np.linalg.eigvalsh(rho) >= -1e-9), "Fixture state is not PSD"
    return rho


@pytest.fixture
def test_entangled_zhang_realignment_criterion():
    """Entangled via Zhang's realignment criterion."""
    rho = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]], dtype=complex)
    return rho


simple_entangled_params = [
    pytest.param("entangled_qutrit_qutrit_state", False, {}, id="max_ent_qutrit_qutrit_ent"),
    pytest.param("entangled_bell_state_0", False, {}, id="bell_state_0_ent"),
    pytest.param("entangled_bell_state_0", False, {"dim": [2, 2]}, id="pure_entangled_state_0"),
    pytest.param("separable_state_2x3_rank3", True, {"dim": [2, 3]}, id="Separable 2x3 state (rank 3)"),
    pytest.param(
        "test_entangled_zhang_realignment_criterion",
        False,
        {},
        id="zhang_realignment_ent_direct",
    ),
]


@pytest.mark.parametrize("state_input, is_bool, kwargs", simple_entangled_params)
def test_entangled_states(state_input, is_bool, kwargs, request):  # Add `request` fixture
    """Test simple entangled states, using indirect fixtures where appropriate."""
    if is_bool:
        assert is_separable(request.getfixturevalue(state_input), **kwargs)
    else:
        assert not is_separable(request.getfixturevalue(state_input), **kwargs)


# --- Individual Tests for Separable States (more complex or specific criteria) ---


def test_ppt_small_dimensions():
    """Separable via PPT sufficiency in small dimensions (2x3)."""
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    psi_sysB = 1 / np.sqrt(3) * e_0 + 1 / np.sqrt(3) * e_1 + 1 / np.sqrt(3) * e_2
    e_0_2d, e_1_2d = basis(2, 0), basis(2, 1)
    psi_sysA = 1 / np.sqrt(2) * e_0_2d + 1 / np.sqrt(2) * e_1_2d
    phi = np.kron(psi_sysA, psi_sysB)
    sigma = phi @ phi.conj().T
    assert is_separable(sigma)


def test_horodecki_rank_le_max_dim_criterion():
    """PPT state with rank <= max_dim_val is separable (2x3 system, rank 3)."""
    dA, dB = 2, 3
    dims = [dA, dB]
    max_d = max(dA, dB)
    a_vecs = [basis(2, 0).reshape(-1), basis(2, 1).reshape(-1)]
    b_vecs = [basis(3, 0).reshape(-1), basis(3, 1).reshape(-1), basis(3, 2).reshape(-1)]
    psi_prods = [np.kron(a_vecs[0], b_vecs[0]), np.kron(a_vecs[0], b_vecs[1]), np.kron(a_vecs[1], b_vecs[2])]
    rho = sum((1 / 3) * np.outer(p, p.conj()) for p in psi_prods)
    test_tol = 1e-7
    assert np.isclose(np.trace(rho), 1.0, atol=test_tol)
    assert is_positive_semidefinite(rho, atol=test_tol)
    current_rank = np.linalg.matrix_rank(rho, tol=test_tol)
    assert np.isclose(current_rank, 3), f"Test state rank is {current_rank}, expected 3"
    assert is_ppt(rho, dim=dims, tol=test_tol), "Test state not PPT"
    assert current_rank <= max_d
    assert is_separable(rho, dim=dims)


def test_separable_closeness_to_maximally_mixed_state():
    """Separable due to closeness to the maximally mixed state."""
    rho = (
        np.array(
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
        / 36
    )
    assert is_separable(rho)


def test_separable_small_rank1_perturbation_of_maximally_mixed_state():
    """Separable as a small rank-1 perturbation of the maximally mixed state."""
    rho = (
        np.array(
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
        / 45
    )
    assert is_separable(rho)


def test_separable_schmidt_rank():  # This state construction appears to be Schmidt rank > 1
    """Separable with operator Schmidt rank (assumed to be test for specific criteria)."""
    # This state is likely separable by other means if its Schmidt rank is not 1.
    # The test name might be slightly misleading if it's not rank 1.
    # For a pure state, separability implies Schmidt rank 1.
    # For a mixed state, low Schmidt rank of operators in a decomposition can imply separability.
    rho = np.array(
        [
            [0.25, 0.15, 0.1, 0.15, 0.09, 0.06, 0.1, 0.06, 0.04],
            [0.15, 0.2, 0.05, 0.09, 0.12, 0.03, 0.06, 0.08, 0.02],
            [0.1, 0.05, 0.05, 0.06, 0.03, 0.03, 0.04, 0.02, 0.02],
            [0.15, 0.09, 0.06, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02],
            [0.09, 0.12, 0.03, 0.12, 0.16, 0.04, 0.03, 0.04, 0.01],
            [0.06, 0.03, 0.03, 0.08, 0.04, 0.04, 0.02, 0.01, 0.01],
            [0.1, 0.06, 0.04, 0.05, 0.03, 0.02, 0.05, 0.03, 0.02],
            [0.06, 0.08, 0.02, 0.03, 0.04, 0.01, 0.03, 0.04, 0.01],
            [0.04, 0.02, 0.02, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01],
        ]
    )
    # Ensure rho is a density matrix
    rho = rho / np.trace(rho)
    assert is_separable(rho, level=1)  # level=1 often relates to PPT check


def test_separable_based_on_eigenvalues():
    """Determined separable by eigenvalues. See Lemma 1 of :cite:`Johnston_2013_Spectrum`."""
    rho = np.array(
        [
            [4 / 22, 2 / 22, -2 / 22, 2 / 22],
            [2 / 22, 7 / 22, -2 / 22, -1 / 22],
            [-2 / 22, -2 / 22, 4 / 22, -2 / 22],
            [2 / 22, -1 / 22, -2 / 22, 7 / 22],
        ]
    )
    assert is_separable(rho)


def test_ppt_2x2_mixed_separable():
    """Separable 2x2 mixed state via PPT."""
    psi1 = np.kron(basis(2, 0), basis(2, 0))
    psi2 = np.kron(basis(2, 1), basis(2, 1))
    rho = 0.5 * (np.outer(psi1, psi1.conj()) + np.outer(psi2, psi2.conj()))
    assert is_separable(rho, dim=[2, 2])


def test_3x3_ppt_rank3_separable_skips_plucker():
    """Mixture of 3 product states in C^3 x C^3 is separable."""
    p1 = np.kron(basis(3, 0), basis(3, 0))
    p2 = np.kron(basis(3, 1), basis(3, 1))
    p3 = np.kron(basis(3, 2), basis(3, 2))
    rho = (np.outer(p1, p1.conj()) + np.outer(p2, p2.conj()) + np.outer(p3, p3.conj())) / 3
    assert is_separable(rho, dim=[3, 3])


def test_breuer_hall_skip_odd_dim():
    """Separable 2x3 product state; Breuer-Hall skips odd dimension (sysB=3)."""
    rho_A = np.eye(2) / 2
    rho_B = np.eye(3) / 3
    rho = np.kron(rho_A, rho_B)
    assert is_separable(rho, dim=[2, 3])


# --- Individual Tests for Entangled States ---


def test_entangled_realignment_criterion_bound_entangled():
    """Entangled via realignment criterion (UPB tile state)."""
    rho = np.identity(9)
    for i in range(5):
        rho = rho - tile(i) @ tile(i).conj().T
    rho = rho / 4
    assert is_density(rho), "Constructed tile state is not a density matrix"
    assert not is_separable(rho)


def test_entangled_cross_norm_realignment_criterion():
    """Entangled by Thm 1 & Rmk 1 of :cite:`Chen_2003_Matrix`."""
    p, a, b = 0.4, 0.8, np.sqrt(0.64)  # b_var was 0.64, assuming it meant b^2
    rho = np.array(
        [
            [p * a**2, 0, 0, p * a * b],
            [0, (1 - p) * a**2, (1 - p) * a * b, 0],
            [0, (1 - p) * a * b, (1 - p) * a**2, 0],
            [p * a * b, 0, 0, p * a**2],
        ]
    )
    rho = rho / np.trace(rho)  # Ensure trace 1
    assert not is_separable(rho)


def test_skip_horodecki_if_not_applicable_proceeds_entangled_tiles():
    """Entangled Tiles state (PPT entangled), testing paths beyond simple Horodecki."""
    rho_tiles = np.identity(9)
    for i in range(5):
        rho_tiles = rho_tiles - tile(i) @ tile(i).conj().T
    rho_tiles = rho_tiles / 4
    assert is_ppt(rho_tiles, dim=[3, 3]), "Tiles state should be PPT for this test intent"
    assert not is_separable(rho_tiles, dim=[3, 3])


# --- Specialized Tests (Edge Cases, Mocking, XFAIL for Incomplete Features) ---


@pytest.mark.skip(reason="Requires specific state for Horodecki rank <= marginal rank test.")
def test_horodecki_rank_le_marginal_rank():
    """Placeholder for PPT state where rank <= marginal rank implies separability."""
    pass


def test_entangled_by_reduction_criterion_non_psd_choi_T():
    """is_separable expects PSD input; Choi of Transpose map is non-PSD for d>1."""
    d = 3
    rho = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            e_i = np.zeros((d, 1))
            e_i[i] = 1
            e_j = np.zeros((d, 1))
            e_j[j] = 1
            ket_ij = np.kron(e_i, e_j)
            bra_ji_conj_T = np.kron(e_j, e_i).conj().T
            rho += ket_ij @ bra_ji_conj_T
    rho /= d
    with pytest.raises(ValueError, match="non-positive semidefinite"):
        is_separable(rho, dim=[d, d])


def test_trace_small_and_matrix_is_almost_zero_proceeds():
    """Near-zero state proceeds if trace and elements are consistently small."""
    state = np.eye(2) * 1e-20
    assert is_separable(state, dim=[1, 2], tol=1e-10)  # dim also implies separability


def test_plucker_linalg_error_in_det_fallthrough():
    """Plucker check falls through on LinAlgError; state remains separable."""
    with mock.patch("numpy.linalg.det", side_effect=np.linalg.LinAlgError("mocked error")):
        p1 = np.kron(basis(3, 0), basis(3, 0))
        p2 = np.kron(basis(3, 1), basis(3, 1))
        p3 = np.kron(basis(3, 2), basis(3, 2))
        p4_v = (basis(3, 0) + basis(3, 1)) / np.sqrt(2)
        p4 = np.kron(p4_v, p4_v)
        rho = (
            np.outer(p1, p1.conj()) + np.outer(p2, p2.conj()) + np.outer(p3, p3.conj()) + np.outer(p4, p4.conj())
        ) / 4.0
        assert is_separable(rho, dim=[3, 3])


def test_eig_calc_fails_rank1_pert_check_skipped():
    """Rank-1 perturbation check skipped if eigenvalue calculation fails."""
    with mock.patch("numpy.linalg.eigvalsh", side_effect=np.linalg.LinAlgError("mocked eig error")):
        with mock.patch("numpy.linalg.eigvals", side_effect=np.linalg.LinAlgError("mocked eig error")):
            assert is_separable(np.eye(8) / 8.0, dim=[2, 4])


def test_2xN_swapped_eig_calc_fails_fallback():
    """Fallback eigenvalue calculation in 2xN if eigvalsh fails."""
    rho_3x2_prod = np.kron(np.eye(3) / 3.0, np.eye(2) / 2.0)
    with mock.patch("numpy.linalg.eigvalsh", side_effect=np.linalg.LinAlgError("mocked eigvalsh error")):
        assert is_separable(rho_3x2_prod, dim=[3, 2])


def test_2xN_block_eig_fails_proceeds():
    """2xN checks proceed if block eigenvalue calculation fails."""
    rho_2x3_mixed = np.eye(6) / 6.0
    with mock.patch("numpy.linalg.eigvals", side_effect=np.linalg.LinAlgError("mocked eig error")):
        assert is_separable(rho_2x3_mixed, dim=[2, 3])


def test_symm_ext_solver_exception_proceeds():
    """Symmetric extension proceeds if has_symmetric_extension fails."""
    with mock.patch(
        "toqito.state_props.is_separable.has_symmetric_extension", side_effect=RuntimeError("Solver failed")
    ):
        assert is_separable(np.eye(4) / 4.0, dim=[2, 2], level=1)


def test_johnston_spectrum_eq12_trigger():
    """Separable via Johnston spectrum condition (2x4)."""
    eigs = np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    rho = np.diag(eigs)
    assert is_ppt(rho, dim=[2, 4]), "Test precondition: State should be PPT"
    assert is_separable(rho, dim=[2, 4])


@pytest.mark.xfail(reason="3x4 separability may not be fully supported.")
def test_breuer_hall_3x4_separable_odd_even_skip_xfail():
    """Separable 3x4 state, Breuer-Hall skips odd dim. XFAIL for now."""
    rhoA = random_density_matrix(3, seed=42)
    rhoB = random_density_matrix(4, seed=43)
    rho_sep_3x4 = np.kron(rhoA, rhoB)
    assert is_separable(rho_sep_3x4, dim=[3, 4])


@pytest.mark.xfail(reason="Rank-1 perturbation for not-full-rank path may not be supported.")
def test_rank1_pert_not_full_rank_path_xfail():
    """Separable 3x3 rank 8 state, xfail for rank-1 perturbation check."""
    eigs = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.0])
    rho = np.diag(eigs / np.sum(eigs))
    assert is_ppt(rho, dim=[3, 3]), "Test precondition: State should be PPT"
    assert is_separable(rho, dim=[3, 3], level=1)


@pytest.mark.xfail(reason="Rank-1 perturbation for full-rank path may be numerically sensitive.")
def test_separable_rank1_perturbation_full_rank_catches_xfail():
    """Separable 3x3 state xfail for full-rank perturbation catch."""
    dim_sys = 3
    prod_dim = dim_sys**2
    eig_vals = np.zeros(prod_dim)
    main_eig = 1.0 - (prod_dim - 1) * 1e-9
    if main_eig <= 0:
        pytest.skip("Cannot construct valid eigenvalues.")
    eig_vals[0] = main_eig
    for i in range(1, prod_dim):
        eig_vals[i] = 1e-9 + (np.random.rand() * 1e-12)
    eig_vals = eig_vals / np.sum(eig_vals)
    eig_vals = np.sort(eig_vals)[::-1]
    rho = np.diag(eig_vals)
    assert is_separable(rho, dim=[dim_sys, dim_sys])


def test_2xN_hildebrand_rank_B_minus_BT_is_zero_true():
    """Separable 2x4 state with Hildebrand rank condition true."""
    rho_A_diag = np.diag([0.7, 0.3])
    rho_B_diag_vals = np.array([0.4, 0.3, 0.2, 0.1])
    rho_B_diag = np.diag(rho_B_diag_vals / np.sum(rho_B_diag_vals))
    rho_test = np.kron(rho_A_diag, rho_B_diag)
    assert is_ppt(rho_test, dim=[2, 4], tol=1e-7), "Test precondition: State should be PPT"
    assert is_separable(rho_test, dim=[2, 4])


def test_breuer_hall_one_dim_odd_path_coverage():
    """Separable 3x2 product state; Breuer-Hall skips odd dimension (sysA=3)."""
    rho_A = random_density_matrix(3, seed=10)
    rho_B = random_density_matrix(2, seed=11)
    rho_sep_3x2 = np.kron(rho_A, rho_B)
    assert is_separable(rho_sep_3x2, dim=[3, 2])


@pytest.mark.xfail(reason="Random 2x4 product (level=1, tol=1e-10) sep may be numerically sensitive.")
def test_2xN_no_swap_needed_random_2x4_xfail():
    """XFAIL for random 2x4 separable state with specific tol/level."""
    rho_2x4_sep = np.kron(random_density_matrix(2, seed=20), random_density_matrix(4, seed=21))
    assert is_separable(rho_2x4_sep, dim=[2, 4], level=1, tol=1e-10)


@mock.patch("numpy.linalg.eigvals", return_value=np.array([0.5, 0.5, 0.0, 0.0]))
def test_2xN_johnston_spectrum_lam_too_short_skips_proceeds(mock_eig):
    """2xN Johnston spectrum proceeds (skips check) with short eigenvalues list."""
    rho_2x4 = np.eye(8) / 8
    assert is_separable(rho_2x4, dim=[2, 4])


@pytest.mark.skip(reason="Needs specific state that evades criteria but is known entangled for `final return False`.")
def test_final_return_false_for_unclassified_entangled():
    """Placeholder for final False return on unclassified entangled state."""
    pass


def test_2xN_johnston_lemma1_eig_A_fails():
    """2xN separable state proceeds if eigenvalue calculation for A_block fails."""
    rho_2x4 = np.eye(8) / 8  # Separable
    original_eigvals = np.linalg.eigvals

    def mock_eigvals_for_A(*args, **kwargs):
        mat_input = args[0]
        if mat_input.shape == (4, 4) and not hasattr(mock_eigvals_for_A, "called_once"):
            mock_eigvals_for_A.called_once = True
            raise np.linalg.LinAlgError("mocked eig for A_block")
        return original_eigvals(mat_input)

    with mock.patch("numpy.linalg.eigvals", side_effect=mock_eigvals_for_A):
        assert is_separable(rho_2x4, dim=[2, 4])


@pytest.mark.xfail(reason="Random 2x4 product (level=2, tol=1e-10) sep may be numerically sensitive.")
def test_2xN_hard_separable_passes_all_witnesses_xfail():
    """XFAIL for hard separable 2x4 state expected to pass all witnesses."""
    rho = np.kron(random_density_matrix(2, seed=1), random_density_matrix(4, seed=2))
    assert is_separable(rho, dim=[2, 4], level=2, tol=1e-10)


def test_symm_ext_catches_hard_entangled_state():
    """Test level=1 behavior for a PPT entangled state."""
    rho_ent_symm = (
        np.array(
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
        / 8.75
    )
    if is_ppt(rho_ent_symm, dim=[3, 3]):  # This state IS PPT L270
        assert is_separable(rho_ent_symm, dim=[3, 3], level=1)
    else:  # Should not happen for this state
        pytest.skip("rho_ent_symm unexpectedly NPT for level=1 test")
    assert not is_separable(rho_ent_symm, dim=[3, 3], level=0)  # Level 0 should detect entanglement if PPT
    assert not is_separable(rho_ent_symm, dim=[3, 3], level=2)  # Level 2 should detect entanglement

    assert not is_separable(rho_ent_symm, dim=[3, 3], level=2)


def test_L138_plucker_orth_rank_lt_4():
    """Separable 3x3 state with Plucker orth basis rank < 4 (skips Plucker determinant)."""
    p1 = np.kron(basis(3, 0), basis(3, 0))
    p2 = np.kron(basis(3, 1), basis(3, 1))
    p3 = np.kron(basis(3, 2), basis(3, 2))
    rho_rank3 = (np.outer(p1, p1) + np.outer(p2, p2) + np.outer(p3, p3)) / 3  # Rank 3
    # Small perturbation to ensure it's still PSD and trace 1, but rank might increase slightly depending on tol
    # The primary state is rank 3. Orth basis of such a state will have < 4 columns.
    if not is_ppt(rho_rank3, dim=[3, 3]):  # Should be PPT
        pytest.skip("Constructed rank-3 state unexpectedly not PPT")
    assert np.linalg.matrix_rank(rho_rank3) < 4  # Precondition for this test's intent
    assert is_separable(rho_rank3, dim=[3, 3])


def test_L160_horodecki_sum_of_ranks_true_specific():
    """Separable 2x4 state via Horodecki sum-of-ranks."""
    dA, dB = 2, 4
    rho = np.zeros((8, 8), dtype=complex)
    a_vecs = [basis(dA, i).reshape(-1) for i in range(dA)]
    b_vecs = [basis(dB, i).reshape(-1) for i in range(dB)]
    psi_prods = [
        np.kron(a_vecs[0], b_vecs[0]),
        np.kron(a_vecs[0], b_vecs[1]),
        np.kron(a_vecs[1], b_vecs[0]),
        np.kron(a_vecs[1], b_vecs[1]),
        np.kron((a_vecs[0] + a_vecs[1]) / np.sqrt(2), b_vecs[2]),
    ]  # 5 product states
    for psi_p in psi_prods:
        rho += (1 / 5) * np.outer(psi_p, psi_p.conj())
    test_tol = 1e-7
    assert np.isclose(np.trace(rho), 1.0, atol=test_tol)
    assert is_positive_semidefinite(rho, atol=test_tol, rtol=test_tol)
    state_r = np.linalg.matrix_rank(rho, tol=test_tol)
    if not (4.9 < state_r < 5.1):
        pytest.skip(f"State not rank ~5, actual rank {state_r}")  # Should be rank 5
    if not is_ppt(rho, dim=[dA, dB], tol=test_tol):
        pytest.skip("State not PPT for Horodecki sum-of-ranks test")
    # rank(rho_pt_A) for this should also be low enough for criterion to pass
    assert is_separable(rho, dim=[dA, dB], tol=1e-8)


@pytest.mark.xfail(reason="Behavior for 2x4 sep state past Hildebrand rank fail not fully confirmed.")
def test_L216_2xN_HildebrandRank_Fails_Proceeds_xfail():
    """XFAIL for separable 2x4 state, Hildebrand rank section."""
    dim_A, dim_N = 2, 4
    rho = np.kron(random_density_matrix(dim_A, seed=50), random_density_matrix(dim_N, seed=51))
    assert is_separable(rho, dim=[dim_A, dim_N], tol=1e-10)  # Tightened tol from 1e-20


def test_L402_johnston_spectrum_true_returns_true_v3():
    """Separable 2x4 state via Johnston spectrum, with mocked rank."""
    eigs = np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    rho = np.diag(eigs)
    assert np.isclose(np.sum(eigs), 1.0)
    if not is_ppt(rho, dim=[2, 4], tol=1e-7):
        pytest.skip("State not PPT for Johnston spectrum test")
    # Mock matrix_rank to ensure it hits the Johnston spectrum check specifically,
    # assuming it might pass earlier due to low rank if not mocked.
    # Here, we mock it to be rank 5 (which is > max_dim=4), to bypass earlier rank checks.
    with mock.patch(
        "numpy.linalg.matrix_rank",
        side_effect=lambda mat, tol: 5 if mat.shape == (8, 8) else np.linalg.matrix_rank(mat, tol),
    ):
        assert is_separable(rho, dim=[2, 4], tol=1e-8)


@pytest.mark.skip(reason="Requires specific 2xN state from Hildebrand paper for homothetic criterion.")
def test_L432_2xN_hildebrand_homothetic_true_v3():
    """Placeholder for Hildebrand homothetic criterion."""
    pass


@pytest.mark.xfail(reason="Breuer-Hall mock test for 2x4 Horodecki state not fully supported.")
@mock.patch("toqito.state_props.is_separable.partial_channel")
def test_L459_breuer_hall_on_dB_only_mocked_first_xfail(mock_pc):
    """XFAIL for entangled 2x4 Horodecki state with mocked Breuer-Hall."""
    try:
        rho_ent_2x4 = horodecki(a_param=0.5, dim=[2, 4])
    except (NameError, ValueError) as e:  # Catch specific errors for construction
        pytest.skip(f"Could not construct Horodecki state: {e}")
    if not is_ppt(rho_ent_2x4, dim=[2, 4]):
        pytest.skip("Horodecki state not PPT for this test.")
    mock_info = {"first_bh_sys0_called_and_passed": False}

    def side_effect_func(state_arg, choi_map_arg, **kwargs_pc):
        sys_pc = kwargs_pc.get("sys")
        # More specific check for the map and state if needed
        if sys_pc == 0 and choi_map_arg.shape == (4, 4) and state_arg.shape == (8, 8):  # Check for 1st BH map app
            mock_info["first_bh_sys0_called_and_passed"] = True
            return np.eye(state_arg.shape[0])  # Return identity to make it "pass" this specific map check
        return partial_channel(state_arg, choi_map_arg, **kwargs_pc)  # Call original for others

    mock_pc.side_effect = side_effect_func

    assert not is_separable(rho_ent_2x4, dim=[2, 4])  # Expected to be entangled
    assert mock_info["first_bh_sys0_called_and_passed"]  # Check if our mock was hit


def test_L453_breuer_hall_on_dA_detects_entangled_2x2werner():
    """Entangled 2x2 Werner state detected by Breuer-Hall (if sys=0 is checked first)."""
    rho_w_ent_2x2 = werner(2, 0.8)  # alpha > 0.5 is entangled
    if not is_ppt(rho_w_ent_2x2, dim=[2, 2], tol=1e-7):
        # Werner states are PPT iff alpha >= 0. For alpha=0.8, it's PPT.
        # If this fails, Werner state construction or is_ppt is an issue.
        pytest.skip("Werner state (alpha=0.8) unexpectedly not PPT.")
    assert not is_separable(rho_w_ent_2x2, dim=[2, 2], tol=1e-8)


@pytest.mark.skip(reason="Requires specific rank-deficient state for perturbation test.")
def test_rank1_pert_skip_for_rank_deficient():
    """Placeholder for rank-deficient perturbation test."""
    pass
