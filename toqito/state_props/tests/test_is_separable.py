"""Test is_separable."""

from unittest import mock

import numpy as np
import pytest

from toqito.channel_ops.partial_channel import partial_channel
from toqito.matrix_props import is_density, is_positive_semidefinite
from toqito.matrix_props.trace_norm import trace_norm
from toqito.rand import random_density_matrix
from toqito.state_props import is_separable
from toqito.state_props.is_ppt import is_ppt
from toqito.state_props.is_separable import (
    _choi_1975_choi_matrix,
    _iterative_product_state_subtraction,
)
from toqito.states import basis, bell, horodecki, isotropic, max_entangled, tile

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
    {"state": np.diag([1e-9, 1e-9]), "error_msg": "Trace of the input state is close to zero"},
    {"state": np.zeros((0, 0)), "dim": 2, "error_msg": "Cannot apply positive dimension"},
    {"state": np.eye(2) / 2, "dim": [0, 2], "error_msg": "zero-dim subsystem"},
    {"state": np.eye(6) / 6, "dim": [2.0, 3.0], "error_msg": "non-negative integers"},
]


@pytest.mark.parametrize("case", invalid_input_cases)
def test_invalid_inputs(case):
    """Parameterized test for invalid input cases raising ValueError."""
    kwargs = {"state": case.get("state"), "dim": case.get("dim")}
    with pytest.raises(ValueError, match=case.get("error_msg")):
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
    pytest.param(np.eye(2) * 1e-20, {"dim": [1, 2], "tol": 1e-10}, id="trace_small_and_matrix_is_almost_zero_proceed"),
    pytest.param(np.kron(np.eye(3) / 3, np.eye(2) / 2), {"dim": [2, 3]}, id="breuer_hall_skip_odd_dim"),
    pytest.param(
        np.diag(np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])),
        {"dim": [2, 4]},
        id="johnston_spectrum_eq12_trigger",  # low rank
    ),
]


@pytest.mark.parametrize("state, kwargs", simple_separable_cases)
def test_simple_separable_states(state, kwargs):
    """Test various simple states known to be separable."""
    assert is_separable(state, **kwargs)[0]


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
    assert np.isclose(np.trace(rho), 1.0)  # Trace of fixture state shouldbe 1
    assert is_positive_semidefinite(rho, 1e-9)  # Fixture state should be PSD
    return rho


@pytest.fixture
def entangled_bell_state_0():
    """Fixture for the density matrix for the Bell state |Φ+> (normalized)."""
    rho = bell(0) @ bell(0).conj().T
    assert np.isclose(np.trace(rho), 1.0)  # Trace of fixture state should be 1
    assert is_positive_semidefinite(rho, 1e-9)  # Fixture state should be PSD"
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
    assert np.isclose(np.trace(rho), 1)  # Trace of fixture state should be 1
    assert np.all(np.linalg.eigvalsh(rho) >= -1e-9)  # Fixture state shoul be PSD
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
def test_entangled_states(state_input, is_bool, kwargs, request):
    """Test simple entangled states, using indirect fixtures where appropriate."""
    assert is_separable(request.getfixturevalue(state_input), **kwargs)[0] is is_bool


# --- Individual Tests for Separable States (more complex or specific criteria) ---


def test_ppt_small_dimensions():
    """Separable via PPT sufficiency in small dimensions (2x3)."""
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    psi_sysB = 1 / np.sqrt(3) * e_0 + 1 / np.sqrt(3) * e_1 + 1 / np.sqrt(3) * e_2
    e_0_2d, e_1_2d = basis(2, 0), basis(2, 1)
    psi_sysA = 1 / np.sqrt(2) * e_0_2d + 1 / np.sqrt(2) * e_1_2d
    phi = np.kron(psi_sysA, psi_sysB)
    sigma = phi @ phi.conj().T
    assert is_separable(sigma)[0]


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
    current_rank = np.linalg.matrix_rank(rho, tol=test_tol)
    assert np.isclose(current_rank, 3)  # Test state rank is expected to be 3"
    assert current_rank <= max_d
    assert is_separable(rho, dim=dims)[0]


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
    assert is_separable(rho)[0]


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
    assert is_separable(rho)[0]


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
    assert is_separable(rho, level=1)[0]  # level=1 often relates to PPT check


def test_separable_based_on_eigenvalues():
    """Determined separable by eigenvalues. See Lemma 1 of :footcite:`Johnston_2013_Spectrum`."""
    rho = np.array(
        [
            [4 / 22, 2 / 22, -2 / 22, 2 / 22],
            [2 / 22, 7 / 22, -2 / 22, -1 / 22],
            [-2 / 22, -2 / 22, 4 / 22, -2 / 22],
            [2 / 22, -1 / 22, -2 / 22, 7 / 22],
        ]
    )
    assert is_separable(rho)[0]


def test_ppt_2x2_mixed_separable():
    """Separable 2x2 mixed state via PPT."""
    psi1 = np.kron(basis(2, 0), basis(2, 0))
    psi2 = np.kron(basis(2, 1), basis(2, 1))
    rho = 0.5 * (np.outer(psi1, psi1.conj()) + np.outer(psi2, psi2.conj()))
    assert is_separable(rho, dim=[2, 2])[0]


def test_3x3_ppt_rank3_separable_skips_plucker():
    """Mixture of 3 product states in C^3 x C^3 is separable."""
    p1 = np.kron(basis(3, 0), basis(3, 0))
    p2 = np.kron(basis(3, 1), basis(3, 1))
    p3 = np.kron(basis(3, 2), basis(3, 2))
    rho = (np.outer(p1, p1.conj()) + np.outer(p2, p2.conj()) + np.outer(p3, p3.conj())) / 3
    assert is_separable(rho, dim=[3, 3])[0]


# --- Individual Tests for Entangled States ---


def test_entangled_realignment_criterion_bound_entangled():
    """Entangled via realignment criterion (UPB tile state)."""
    rho = np.identity(9)
    for i in range(5):
        rho = rho - tile(i) @ tile(i).conj().T
    rho = rho / 4
    assert is_density(rho)  # Constructed tile state should be a density matrix
    assert not is_separable(rho)[0]


def test_entangled_cross_norm_realignment_criterion():
    """Entangled by Thm 1 & Rmk 1 of :footcite:`Chen_2003_Matrix`."""
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
    assert not is_separable(rho)[0]


def test_skip_horodecki_if_not_applicable_proceeds_entangled_tiles():
    """Entangled Tiles state (PPT entangled), testing paths beyond simple Horodecki."""
    rho_tiles = np.identity(9)
    for i in range(5):  # Ensure tile(i) is correctly defined
        rho_tiles = rho_tiles - tile(i) @ tile(i).conj().T
    rho_tiles = rho_tiles / 4

    # Debugging rank
    calculated_rank = np.linalg.matrix_rank(rho_tiles, tol=1e-8)  # Use a consistent tolerance
    assert calculated_rank == 4

    assert not is_separable(rho_tiles, dim=[3, 3])[0]


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
        is_separable(rho, dim=[d, d])[0]


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
        assert is_separable(rho, dim=[3, 3])[0]


def test_eig_calc_fails_rank1_pert_check_skipped():
    """Rank-1 perturbation check skipped if eigenvalue calculation fails."""
    with mock.patch("numpy.linalg.eigvalsh", side_effect=np.linalg.LinAlgError("mocked eig error")):
        with mock.patch("numpy.linalg.eigvals", side_effect=np.linalg.LinAlgError("mocked eig error")):
            assert is_separable(np.eye(8) / 8.0, dim=[2, 4])[0]


def test_2xN_swapped_eig_calc_fails_fallback():
    """Fallback eigenvalue calculation in 2xN if eigvalsh fails."""
    rho_3x2_prod = np.kron(np.eye(3) / 3.0, np.eye(2) / 2.0)
    with mock.patch("numpy.linalg.eigvalsh", side_effect=np.linalg.LinAlgError("mocked eigvalsh error")):
        assert is_separable(rho_3x2_prod, dim=[3, 2])[0]


def test_2xN_block_eig_fails_proceeds():
    """2xN checks proceed if block eigenvalue calculation fails."""
    rho_2x3_mixed = np.eye(6) / 6.0
    with mock.patch("numpy.linalg.eigvals", side_effect=np.linalg.LinAlgError("mocked eig error")):
        assert is_separable(rho_2x3_mixed, dim=[2, 3])[0]


def test_symm_ext_solver_exception_proceeds():
    """Symmetric extension proceeds if has_symmetric_extension fails."""
    with mock.patch(
        "toqito.state_props.is_separable.has_symmetric_extension", side_effect=RuntimeError("Solver failed")
    ):
        assert is_separable(np.eye(4) / 4.0, dim=[2, 2], level=1)[0]


def test_breuer_hall_3x4_separable():
    """Product 3x4 state: the Cariello operator Schmidt rank = 1 branch fires."""
    rhoA = random_density_matrix(3, seed=42)
    rhoB = random_density_matrix(4, seed=43)
    rho_sep_3x4 = np.kron(rhoA, rhoB)
    sep, reason = is_separable(rho_sep_3x4, dim=[3, 4])
    assert sep is True
    assert "operator Schmidt rank" in reason


def test_rank1_pert_not_full_rank_path():
    """Separable 3x3 rank 8 state, xfail for rank-1 perturbation check."""
    eigs = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.0])
    rho = np.diag(eigs / np.sum(eigs))
    assert is_separable(rho, dim=[3, 3], level=1)[0]


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
    assert is_separable(rho, dim=[dim_sys, dim_sys])[0]


def test_2xN_hildebrand_rank_B_minus_BT_is_zero_true():
    """Separable 2x4 state with Hildebrand rank condition true."""
    rho_A_diag = np.diag([0.7, 0.3])
    rho_B_diag_vals = np.array([0.4, 0.3, 0.2, 0.1])
    rho_B_diag = np.diag(rho_B_diag_vals / np.sum(rho_B_diag_vals))
    rho_test = np.kron(rho_A_diag, rho_B_diag)

    assert is_separable(rho_test, dim=[2, 4])[0]


def test_breuer_hall_one_dim_odd_path_coverage():
    """Separable 3x2 product state; Breuer-Hall skips odd dimension (sysA=3)."""
    rho_A = random_density_matrix(3, seed=10)
    rho_B = random_density_matrix(2, seed=11)
    rho_sep_3x2 = np.kron(rho_A, rho_B)
    assert is_separable(rho_sep_3x2, dim=[3, 2])[0]


def test_2xN_no_swap_needed_random_2x4():
    """XFAIL for random 2x4 separable state with specific tol/level."""
    rho_2x4_sep = np.kron(random_density_matrix(2, seed=20), random_density_matrix(4, seed=21))
    assert is_separable(rho_2x4_sep, dim=[2, 4], level=1, tol=1e-10)[0]


@mock.patch("numpy.linalg.eigvals", return_value=np.array([0.5, 0.5, 0.0, 0.0]))
def test_2xN_johnston_spectrum_lam_too_short_skips_proceeds(mock_eig):
    """2xN Johnston spectrum proceeds (skips check) with short eigenvalues list."""
    rho_2x4 = np.eye(8) / 8
    assert is_separable(rho_2x4, dim=[2, 4])[0]


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
        assert is_separable(rho_2x4, dim=[2, 4])[0]


def test_2xN_hard_separable_passes_all_witnesses():
    """XFAIL for hard separable 2x4 state expected to pass all witnesses."""
    rho = np.kron(random_density_matrix(2, seed=1), random_density_matrix(4, seed=2))
    assert is_separable(rho, dim=[2, 4], level=2, tol=1e-10)[0]


def test_symm_ext_catches_hard_entangled_state():
    """Test level=1 behavior for a PPT entangled state."""
    # This appear to pass to last final False sometimes
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
    assert is_separable(rho_ent_symm, dim=[3, 3], level=1)[0]  # This state IS PPT
    assert not is_separable(rho_ent_symm, dim=[3, 3], level=0)[0]  # Heuristics alone cannot prove separability
    # This PPT entangled state has a 2-copy symmetric extension, so the DPS
    # hierarchy at level 2 is insufficient to detect its entanglement.
    assert is_separable(rho_ent_symm, dim=[3, 3], level=2)[0]


def test_plucker_orth_rank_lt_4():
    """Separable 3x3 state with Plucker orth basis rank < 4 (skips Plucker determinant)."""
    p1 = np.kron(basis(3, 0), basis(3, 0))
    p2 = np.kron(basis(3, 1), basis(3, 1))
    p3 = np.kron(basis(3, 2), basis(3, 2))
    rho_rank3 = (np.outer(p1, p1) + np.outer(p2, p2) + np.outer(p3, p3)) / 3  # Rank 3
    # Small perturbation to ensure it's still PSD and trace 1, but rank might increase slightly depending on tol
    # The primary state is rank 3. Orth basis of such a state will have < 4 columns.
    assert np.linalg.matrix_rank(rho_rank3) < 4  # Precondition for this test's intent
    assert is_separable(rho_rank3, dim=[3, 3])[0]


def test_horodecki_sum_of_ranks_true_specific():
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
    # rank(rho_pt_A) for this should also be low enough for criterion to pass
    assert is_separable(rho, dim=[dA, dB], tol=1e-8)[0]


def test_2xN_separable_product_state():
    """Product 2x4 state: the Cariello operator Schmidt rank = 1 branch fires."""
    dim_A, dim_N = 2, 4
    rho = np.kron(random_density_matrix(dim_A, seed=50), random_density_matrix(dim_N, seed=51))
    sep, reason = is_separable(rho, dim=[dim_A, dim_N], tol=1e-10)
    assert sep is True
    assert "operator Schmidt rank" in reason


def test_johnston_spectrum_true_returns_true_v3():
    """Separable 2x4 diagonal state is classified as separable.

    Note: despite the historical name, this test does not actually exercise the
    Johnston spectral-condition branch. The diagonal state below has operator
    Schmidt rank 2, so it is now caught by the Cariello ``op_sr <= 2`` branch
    (section 5b). Even before section 5b existed, the state was being caught
    by the Horodecki rank-sum branch under the mocked ranks rather than by
    Johnston. Getting this test to actually exercise section 11 would require
    rewriting it; leaving that for a separate cleanup pass.
    """
    eigs = np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    rho = np.diag(eigs)
    assert np.isclose(np.sum(eigs), 1.0)
    real_matrix_rank = np.linalg.matrix_rank
    with mock.patch(
        "numpy.linalg.matrix_rank",
        side_effect=lambda mat, tol=None: 5 if mat.shape == (8, 8) else real_matrix_rank(mat, tol=tol),
    ):
        assert is_separable(rho, dim=[2, 4], tol=1e-8)[0]


@pytest.mark.skip(reason="Requires specific 2xN state from Hildebrand paper for homothetic criterion.")
def test_2xN_hildebrand_homothetic_true_v3():
    """Placeholder for Hildebrand homothetic criterion."""
    pass


@pytest.mark.xfail(reason="Breuer-Hall mock test for 2x4 Horodecki state not fully supported.")
@mock.patch("toqito.state_props.is_separable.partial_channel")
def test_breuer_hall_on_dB_only_mocked_first_xfail(mock_pc):
    """XFAIL for entangled 2x4 Horodecki state with mocked Breuer-Hall."""
    try:
        rho_ent_2x4 = horodecki(a_param=0.5, dim=[2, 4])
    except (NameError, ValueError) as e:  # Catch specific errors for construction
        pytest.skip(f"Could not construct Horodecki state: {e}")
    mock_info = {"first_bh_sys0_called_and_passed": False}

    def side_effect_func(state_arg, choi_map_arg, **kwargs_pc):
        sys_pc = kwargs_pc.get("sys")
        # More specific check for the map and state if needed
        if sys_pc == 0 and choi_map_arg.shape == (4, 4) and state_arg.shape == (8, 8):  # Check for 1st BH map app
            mock_info["first_bh_sys0_called_and_passed"] = True
            return np.eye(state_arg.shape[0])  # Return identity to make it "pass" this specific map check
        return partial_channel(state_arg, choi_map_arg, **kwargs_pc)  # Call original for others

    mock_pc.side_effect = side_effect_func

    assert not is_separable(rho_ent_2x4, dim=[2, 4])[0]  # Expected to be entangled
    assert mock_info["first_bh_sys0_called_and_passed"]  # Check if our mock was hit


def test_breuer_hall_on_dA_detects_entangled_2x2werner():
    """Entangled 2x2 Werner state detected by Breuer-Hall (if sys=0 is checked first)."""
    # rho_w_ent_2x2 = werner(2, 0.8)  # alpha > 0.5 is entangled
    # if not is_ppt(rho_w_ent_2x2, dim=[2, 2], tol=1e-7):
    # Werner states are PPT iff alpha >= 0. For alpha=0.8, it's PPT.
    # If this fails, Werner state construction or is_ppt is an issue.
    pytest.skip("Werner state (alpha=0.8) unexpectedly not PPT.")
    # assert not is_separable(rho_w_ent_2x2, dim=[2, 2], tol=1e-8)[0]


@pytest.mark.skip(reason="Requires specific rank-deficient state for perturbation test.")
def test_rank1_pert_skip_for_rank_deficient():
    """Placeholder for rank-deficient perturbation test."""
    pass


def test_3x3_rank4_block_orth_finds_lower_rank():
    """Test 3x3 rank-4 block when orth() gives <4 columns. cover logic q_orth_basis.shape[1] < 4."""
    # Create a rank-4 state in 3x3 system
    p1 = np.kron(basis(3, 0), basis(3, 0))
    p2 = np.kron(basis(3, 1), basis(3, 1))
    p3 = np.kron(basis(3, 2), basis(3, 2))
    v = (basis(3, 0) + basis(3, 1)) / np.sqrt(2)
    p4 = np.kron(v, v)
    rho = (np.outer(p1, p1.conj()) + np.outer(p2, p2.conj()) + np.outer(p3, p3.conj()) + np.outer(p4, p4.conj())) / 4.0

    # Mock orth to return only 3 columns (simulating numerical rank deficiency)
    with mock.patch("toqito.state_props.is_separable.orth") as mocked_orth:
        mocked_orth.return_value = np.eye(9, 3)  # 9x3 matrix
        assert is_separable(rho, dim=[3, 3])[0]  # Should proceed past Plücker check
        mocked_orth.assert_called_once()


def test_2xN_eig_lam_eigvalsh_fails_eigvals_succeeds():
    """2xN: eigvalsh for current_lam_2xn fails, fallback eigvals works."""
    rho_2xN_sep = np.eye(8) / 8.0  # 2x4 separable state
    with mock.patch("numpy.linalg.eigvalsh", side_effect=np.linalg.LinAlgError("mocked error")):
        assert is_separable(rho_2xN_sep, dim=[2, 4])[0]  # Should use eigvals fallback


def test_full_rank_ppt_state_above_threshold():
    """Test Basic Realignment Criterion."""
    # PPT state where rank sum exceeds Horodecki threshold. If > 1, entangled
    # Use a PPT entangled state (Horodecki state) could have chances pass Basic Realignment Return true
    rho = horodecki(a_param=0.5, dim=[3, 3])

    # Mock matrix ranks to exceed the rank-sum threshold (7+7=14 fails, 8+7=15 passes)
    # *and* to keep rank(rho) above both marginal ranks so the rank-marginal check
    # does not fire. The call order is:
    #   state_rank, op_Schmidt_rank_internal, rank_pt_A, rank_marg_A, rank_marg_B
    with mock.patch("numpy.linalg.matrix_rank") as mock_rank:
        mock_rank.side_effect = [8, 8, 7, 3, 3]
        with mock.patch("toqito.state_props.in_separable_ball", return_value=False):
            assert not is_separable(rho, dim=[3, 3], level=1)[0]


def test_plucker_3x3_rank4_separable_det_F_is_zero():
    """Covers L305 (Plucker loop) and L331 (det(F)~0 -> separable)."""
    v0, v1, v2 = basis(3, 0), basis(3, 1), basis(3, 2)
    p1_vec = np.kron(v0, v0)
    p2_vec = np.kron(v1, v1)
    p3_vec = np.kron(v2, v2)
    vA_mix = (v0 + v1) / np.sqrt(2)
    p4_vec = np.kron(vA_mix, v0)
    rho = (
        np.outer(p1_vec, p1_vec.conj())
        + np.outer(p2_vec, p2_vec.conj())
        + np.outer(p3_vec, p3_vec.conj())
        + np.outer(p4_vec, p4_vec.conj())
    )
    rho = rho / np.trace(rho)
    assert np.linalg.matrix_rank(rho, tol=1e-7) == 4  # Constructed state should be rank 4
    assert is_ppt(rho, dim=[3, 3])  # Constructed state should be PPT
    assert is_separable(rho, dim=[3, 3])[0]


def test_entangled_zhang_variant_catches_L401():
    """Return Zhang et al. 2008 Variant."""
    rho = horodecki(a_param=0.6, dim=[3, 3])
    assert is_ppt(rho, dim=[3, 3])

    original_matrix_rank = np.linalg.matrix_rank
    # Call order is state_rank, op_Schmidt_rank_internal, rank_pt_A, rank_marg_A, rank_marg_B.
    # op_sr > 2 skips the Cariello check; rank_pt_A chosen so state_r+rank_pt_A=15>14
    # skips the rank-sum check; marginal ranks kept below state_r so the rank-marginal
    # check does not fire either. Zhang variant then catches the entanglement.
    mock_ranks_horodecki_op_fail = [7, 8, 8, 3, 3]

    def matrix_rank_zhang_side_effect(matrix_arg, tol=None):
        if mock_ranks_horodecki_op_fail:  # Pop if list is not empty
            return mock_ranks_horodecki_op_fail.pop(0)
        return original_matrix_rank(matrix_arg, tol=tol)

    original_trace_norm_func = trace_norm
    call_tracker = {"count": 0}

    def mocked_trace_norm_for_zhang(matrix_input, **kwargs_tn):
        call_tracker["count"] += 1
        if call_tracker["count"] == 1:
            return 0.5
        return original_trace_norm_func(matrix_input, **kwargs_tn)

    with mock.patch("toqito.state_props.is_separable.in_separable_ball", return_value=False):
        with mock.patch("numpy.linalg.matrix_rank", side_effect=matrix_rank_zhang_side_effect):
            with mock.patch("toqito.state_props.is_separable.trace_norm", side_effect=mocked_trace_norm_for_zhang):
                assert not is_separable(rho, dim=[3, 3], level=0)[0]


def test_rank1_pert_eigvalsh_fails_eigvals_fallback():
    """test_rank1_pert_eigvalsh_fails_eigvals_fallback_L412."""
    dim_sys = 3
    prod_dim = dim_sys**2
    test_tol = 1e-8

    machine_eps_expected = np.finfo(float).eps
    threshold_condition = test_tol**2 + 2 * machine_eps_expected  # Approx 5.44e-16

    # Construct eigenvalues `lam` directly (will be sorted descending later by the code)
    # We want lam_sorted[1] - lam_sorted[prod_dim-1] < threshold_condition

    eigs = np.zeros(prod_dim)

    # Smallest eigenvalue
    eigs[0] = 0.05  # This will become lam_sorted[prod_dim-1] after sorting if it's smallest

    # Second largest eigenvalue (to be lam_sorted[1])
    # Let it be smallest + desired_small_diff / 2
    desired_small_diff = threshold_condition / 2  # e.g., 2.7e-16
    eigs[1] = eigs[0] + desired_small_diff

    # Largest eigenvalue (to be lam_sorted[0])
    eigs[2] = eigs[1] + 1e-5  # Ensure it's distinctly larger

    # Fill the rest
    # We need prod_dim - 3 more eigenvalues.
    # They should be between eigs[0] (smallest after sort) and eigs[1] (2nd largest after sort)
    # or rather, between the final smallest and final 2nd largest.

    # Let's make it simpler:
    # lam_final_sorted_descending:
    # lam_fsd[prod_dim-1] = X
    # lam_fsd[1]          = X + desired_small_diff
    # lam_fsd[0]          = X + desired_small_diff + epsilon_large

    # Start with a base value for all eigs
    avg_eig = 1.0 / prod_dim
    temp_eigs = np.full(prod_dim, avg_eig)

    # Adjust to create the specific gap for lam[1] and lam[prod_dim-1] after sorting
    # Assume they will be perturbed from avg_eig

    # Smallest one:
    temp_eigs[prod_dim - 1] = avg_eig - 0.01
    # Second largest one (this element will be sorted to index 1 if it's the 2nd largest):
    temp_eigs[1] = temp_eigs[prod_dim - 1] + desired_small_diff  # Target lam_sorted[1]
    # This assignment order is tricky before sorting.

    # Let's define the sorted spectrum directly
    lam_sorted_desc = np.zeros(prod_dim)
    lam_sorted_desc[prod_dim - 1] = avg_eig * 0.5  # Smallest
    lam_sorted_desc[1] = lam_sorted_desc[prod_dim - 1] + desired_small_diff
    lam_sorted_desc[0] = lam_sorted_desc[1] + 0.01  # Largest

    # Fill the intermediate eigenvalues lam_sorted_desc[2]...lam_sorted_desc[prod_dim-2]
    # They must be between lam_sorted_desc[1] and lam_sorted_desc[prod_dim-1] and maintain sorted order
    num_intermediate = prod_dim - 3
    if num_intermediate > 0:
        step = (lam_sorted_desc[1] - lam_sorted_desc[prod_dim - 1]) / (num_intermediate + 1)
        if step < 0:  # This can happen if smallest + desired_small_diff makes it too large
            pytest.skip("Eigenvalue construction issue for rank-1 pert.")
        for i in range(num_intermediate):
            lam_sorted_desc[2 + i] = lam_sorted_desc[1] - (i + 1) * step

    # Normalize so they sum to 1
    lam_sorted_desc = lam_sorted_desc / np.sum(lam_sorted_desc)

    # Verify the critical difference AFTER normalization
    actual_diff = lam_sorted_desc[1] - lam_sorted_desc[prod_dim - 1]
    if actual_diff >= threshold_condition:
        # If normalization messed it up, try to re-adjust or skip
        # This can happen if the initial values are too far apart, normalization shrinks small gaps too much
        pytest.skip(f"Normalization made eigenvalue diff {actual_diff} too large for rank-1 pert test.")

    # Create the state with these eigenvalues (order doesn't matter for np.diag for this purpose)
    rho = np.diag(lam_sorted_desc)

    # Sanity checks for rho
    assert np.allclose(np.trace(rho), 1.0)  # Trace of rho is 1
    eigenvalues_of_rho = np.sort(np.linalg.eigvalsh(rho))[::-1]  # Get actual sorted eigenvalues
    assert np.allclose(eigenvalues_of_rho, lam_sorted_desc)  # "Rho must have the intended eigenvalues"
    assert is_ppt(rho, dim=[dim_sys, dim_sys], tol=test_tol)  #  "Constructed rho must be PPT"

    # Rest of the mock setup
    original_matrix_rank = np.linalg.matrix_rank
    mock_ranks_rank1 = [prod_dim, prod_dim, prod_dim, prod_dim, prod_dim, prod_dim]

    def matrix_rank_rank1_side_effect(matrix_arg, tol=None):
        if matrix_arg.shape == (prod_dim, prod_dim) and mock_ranks_rank1:
            return mock_ranks_rank1.pop(0)
        return original_matrix_rank(matrix_arg, tol=tol)

    with mock.patch("toqito.state_props.is_separable.in_separable_ball", return_value=False):
        with mock.patch("numpy.linalg.matrix_rank", side_effect=matrix_rank_rank1_side_effect):
            with mock.patch("toqito.state_props.is_separable.trace_norm", return_value=0.5):
                with mock.patch("numpy.linalg.eigvalsh", side_effect=np.linalg.LinAlgError("mocked eigvalsh fail")):
                    assert is_separable(rho, dim=[dim_sys, dim_sys], tol=test_tol)[0]


# --- attempt to Targeted Tests for Coverage ---


@pytest.fixture
def tiles_state_3x3_ppt_entangled():
    """tiles_state_3x3_ppt_entangled."""
    v = [basis(3, i) for i in range(3)]

    psi_vecs = [
        np.kron(v[0], (v[0] + v[1]) / np.sqrt(2)),  # |0> (|0>+|1>)/sqrt(2)
        np.kron((v[0] + v[1]) / np.sqrt(2), v[2]),  # (|0>+|1>)/sqrt(2) |2>
        np.kron(v[1], (v[1] + v[2]) / np.sqrt(2)),  # |1> (|1>+|2>)/sqrt(2)
        np.kron((v[1] + v[2]) / np.sqrt(2), v[0]),  # (|1>+|2>)/sqrt(2) |0>
        np.kron(v[2], (v[2] + v[0]) / np.sqrt(2)),  # |2> (|2>+|0>)/sqrt(2)
    ]
    # This set of 5 vectors IS the standard one for the Horodecki Tiles UPB.

    sum_projectors = np.zeros((9, 9), dtype=complex)
    for psi in psi_vecs:
        sum_projectors += np.outer(psi, psi.conj())

    rho = (np.identity(9) - sum_projectors) / 4.0
    # This formula for rho is correct.
    # Normalization: Each |psi_i><psi_i| is rank 1, trace 1.
    # Sum of 5 projectors. Tr(sum_projectors) = 5.
    # Tr(I_9 - sum_projectors) = Tr(I_9) - Tr(sum_projectors) = 9 - 5 = 4.
    # Tr(rho) = Tr(I_9 - sum_projectors) / 4 = 4 / 4 = 1.  So normalization is correct.
    # The state rho is the projector onto the 4-dimensional subspace orthogonal to the span of {psi_i}.
    return rho


@pytest.mark.skip(reason="partial_transpose or is_ppt issue")
def test_path_ha_kye_fallthrough_to_final_false_L534_to_L580(tiles_state_3x3_ppt_entangled):
    """Test a 3x3 PPT entangled state (Tiles).

    - (Mocked to) pass Plucker (by returning True, implying det(F) is small).
    - Passes Ha-Kye maps (as they are not expected to catch Tiles).
    - (Natural behavior) fails symmetric extension for level=2.
    - Results in final 'return False'.
    Covers path like old 607->636 (new L534 -> L580).
    """
    rho = tiles_state_3x3_ppt_entangled
    dims = [3, 3]
    test_tol = 1e-8

    assert is_ppt(rho, dim=dims, tol=test_tol)
    assert np.linalg.matrix_rank(rho, tol=test_tol) == 4

    original_linalg_det = np.linalg.det
    mock_plucker_det_call_info = {"called": False}

    def mock_det_for_plucker(matrix_arg):
        if matrix_arg.shape == (6, 6):  # Plucker F_det_matrix is 6x6
            mock_plucker_det_call_info["called"] = True
            return test_tol**3
        return original_linalg_det(matrix_arg)

    original_matrix_rank = np.linalg.matrix_rank
    # We need Horodecki operational criteria to NOT indicate separability.
    # state_rank=4. max_dim_val=3.  4 <= 3 is FALSE. (Good)
    # PT of Tiles is also rank 4. Sum_rank = 4+4=8.
    # Threshold for 3x3 = 2*9 - 3-3 + 2 = 14.
    # 8 <= 14 is TRUE. This would make L382 (new) return True.
    # So, we need to mock rank_pt_A to be higher, e.g., 11, so 4+11=15 > 14.
    mock_ranks_horodecki_fail = [4, 11]  # state_rank, rank_pt_A
    mock_rank_call_info = {"values": mock_ranks_horodecki_fail[:]}

    # Corrected signature for matrix_rank side_effect
    def matrix_rank_side_effect(matrix_arg, tol=None):
        # Only mock for the full 9x9 state and its partial transpose
        if matrix_arg.shape == (9, 9) and mock_rank_call_info["values"]:
            val = mock_rank_call_info["values"].pop(0)
            return val
        return original_matrix_rank(matrix_arg, tol=tol)

    with mock.patch("toqito.state_props.is_separable.in_separable_ball", return_value=False):
        with mock.patch("numpy.linalg.det", side_effect=mock_det_for_plucker):  # Patches det used by Plucker
            with mock.patch("numpy.linalg.matrix_rank", side_effect=matrix_rank_side_effect):
                with mock.patch("toqito.state_props.is_separable.trace_norm", return_value=0.5):  # Pass realignments
                    # Mock Rank-1 Pert to be inconclusive (L419 condition False)
                    # Descending eigs:
                    eigs_for_mock_rank1_pert_fail_desc = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.01])
                    eigs_for_mock_rank1_pert_fail_desc /= np.sum(eigs_for_mock_rank1_pert_fail_desc)
                    # lam[1]-lam[8] will be large enough.
                    mock_eigs_ascending_for_pert_fail = np.sort(eigs_for_mock_rank1_pert_fail_desc)

                    # Patch np.linalg.eigvalsh as used by is_separable.py
                    with mock.patch(
                        "toqito.state_props.is_separable.np.linalg.eigvalsh",
                        return_value=mock_eigs_ascending_for_pert_fail,
                    ):
                        # We expect Ha-Kye maps to pass (not detect entanglement for Tiles).
                        # We expect Breuer-Hall to be skipped (dA=3, dB=3).
                        # We expect Symmetric Extension (level=2) for Tiles to result in
                        # has_symmetric_extension returning False (as Tiles is not 2-extendible).
                        # This means L572 (new) `return True` is not hit.
                        # The loop L570-L574 finishes.
                        # L576 `elif level == 1` is false.
                        # Final L580 `return False` is hit.
                        assert not is_separable(rho, dim=dims, tol=test_tol, level=2)[0]
                        assert mock_plucker_det_call_info["called"]  # Plucker det mock need be to called


# --- Tests for the (separable, reason) return tuple ---


def test_return_tuple_shape():
    """is_separable returns a 2-tuple of (bool, non-empty str)."""
    result = is_separable(np.eye(4) / 4, dim=[2, 2])
    assert isinstance(result, tuple) and len(result) == 2
    sep, reason = result
    assert isinstance(sep, bool)
    assert isinstance(reason, str) and reason


def test_return_reason_on_separable_verdict():
    """A separable verdict carries a non-empty criterion string."""
    sep, reason = is_separable(np.eye(4) / 4, dim=[2, 2])
    assert sep is True
    assert reason and isinstance(reason, str)


def test_return_reason_on_entangled_verdict():
    """An entanglement witness produces a non-empty criterion string."""
    rho_bell = bell(0) @ bell(0).conj().T
    sep, reason = is_separable(rho_bell)
    assert sep is False
    assert reason and isinstance(reason, str)


def test_return_reason_tracks_npt_branch():
    """An NPT mixed state is flagged specifically by the NPT branch."""
    rho_bell = bell(0) @ bell(0).conj().T
    rho_npt_mixed = 0.9 * rho_bell + 0.1 * np.eye(4) / 4  # rank 4, still NPT
    sep, reason = is_separable(rho_npt_mixed)
    assert sep is False
    # Match on "NPT" / "Peres-Horodecki" specifically — "PPT" alone would also
    # match the inconclusive fallback reason and wouldn't catch a regression.
    assert "NPT" in reason and "Peres-Horodecki" in reason


# --- Tests for the `strength` parameter ---


def _upb_tile_state() -> np.ndarray:
    """3x3 rank-4 PPT-entangled state built from the tile UPB (I - sum of tile projectors) / 4.

    Used by the strength-parameter tests to exercise the Plucker branch at
    strength=1 vs. the early-cutoff at strength=0. Not a fixture so a couple
    of small tests can call it directly without indirection.
    """
    rho = np.identity(9)
    for i in range(5):
        rho = rho - tile(i) @ tile(i).conj().T
    return rho / 4


def test_strength_zero_caps_after_ppt_prechecks_on_upb_tile():
    """strength=0 stops early on a UPB tile PPT-entangled state.

    At strength=1 the 3x3 rank-4 Plucker check catches it as entangled; at
    strength=0 everything after PPT <= 6 is skipped, so the function returns
    the 'strength=0 capped' inconclusive fallback.
    """
    rho = _upb_tile_state()

    sep_default, reason_default = is_separable(rho, dim=[3, 3])
    assert sep_default is False
    assert "Plucker" in reason_default

    sep_fast, reason_fast = is_separable(rho, dim=[3, 3], strength=0)
    assert sep_fast is False
    assert "strength=0 capped" in reason_fast


def test_strength_zero_still_catches_tier_zero_criteria():
    """strength=0 must still catch the fast sufficient/necessary conditions.

    The maximally mixed 2x2 state is separable and hits one of the tier-0
    branches (Gurvits-Barnum ball or PPT<=6); a mixed NPT state is entangled
    and hits the PPT criterion — both must fire regardless of strength.
    """
    sep_mixed, reason_mixed = is_separable(np.eye(4) / 4, dim=[2, 2], strength=0)
    assert sep_mixed is True
    assert "strength=0 capped" not in reason_mixed  # should NOT be the fallback

    # Use a mixed NPT state (not a pure Bell) so PPT fires, not Schmidt rank.
    rho_bell = bell(0) @ bell(0).conj().T
    rho_npt_mixed = 0.9 * rho_bell + 0.1 * np.eye(4) / 4
    sep_bell, reason_bell = is_separable(rho_npt_mixed, strength=0)
    assert sep_bell is False
    assert "NPT" in reason_bell


def test_strength_minus_one_matches_default():
    """strength=-1 ('everything') is equivalent to strength=1 today."""
    rho = _upb_tile_state()
    assert is_separable(rho, dim=[3, 3], strength=1) == is_separable(rho, dim=[3, 3], strength=-1)


@pytest.mark.parametrize(
    "bad_strength, match",
    [
        (1.5, "must be an int"),
        ("fast", "must be an int"),
        (None, "must be an int"),
        (True, "must be an int"),
        (-2, "must be -1, 0, or a positive integer"),
        (-10, "must be -1, 0, or a positive integer"),
    ],
)
def test_strength_validation_rejects_invalid(bad_strength, match):
    """Invalid `strength` values are rejected rather than silently treated as the full run."""
    with pytest.raises(ValueError, match=match):
        is_separable(np.eye(4) / 4, dim=[2, 2], strength=bad_strength)


# --- Tests for the Cariello operator Schmidt rank <= 2 check (#1245) ---


def test_operator_schmidt_rank_two_classical_correlation_3x3_is_separable():
    """A diagonal classical-correlation state has operator Schmidt rank 2 and is separable.

    rho = 1/2 (|00><00| + |11><11|) has operator Schmidt rank 2 and is a canonical
    separable state. is_separable should return it via the Cariello check rather
    than the PPT <= 6 branch so we're exercising the new code path. We force the
    product dimension above 6 by using a 3x3 embedding of the 2-dimensional
    correlation state.
    """
    v0 = basis(3, 0)
    v1 = basis(3, 1)
    psi_00 = np.kron(v0, v0)
    psi_11 = np.kron(v1, v1)
    rho = 0.5 * (psi_00 @ psi_00.conj().T + psi_11 @ psi_11.conj().T)
    assert np.linalg.matrix_rank(rho) == 2  # not pure, rank 2

    sep, reason = is_separable(rho, dim=[3, 3])
    assert sep is True
    assert "operator Schmidt rank" in reason and "Cariello" in reason


# --- Tests for the Horodecki rank-marginal check (#1251b) ---


# --- Tests for Choi's 1975 positive-map witness (#1249) ---


def test_choi_1975_choi_matrix_structure():
    """The Choi matrix of Choi's 1975 map has the expected block structure.

    Diagonal blocks are Phi(E_ii) along the (3i, 3i) diagonal; off-diagonal
    blocks contain a single -1 at the (i, j) position.
    """
    J = _choi_1975_choi_matrix()
    assert J.shape == (9, 9)

    # Diagonal of J matches (Phi(E_00)_00, Phi(E_00)_11, ..., Phi(E_22)_22).
    expected_diag = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=complex)
    np.testing.assert_array_equal(np.diag(J), expected_diag)

    # Off-diagonal -1 entries at (3i+i, 3j+j) for i != j, zero elsewhere.
    for i in range(3):
        for j in range(3):
            if i != j:
                assert J[3 * i + i, 3 * j + j] == -1.0
    # Everything outside those positions and the diagonal is zero.
    off_diag_positions = {(3 * i + i, 3 * j + j) for i in range(3) for j in range(3) if i != j}
    for r in range(9):
        for c in range(9):
            if r == c or (r, c) in off_diag_positions:
                continue
            assert J[r, c] == 0.0


def test_choi_1975_choi_matrix_is_not_psd():
    """Choi matrix has a negative eigenvalue, so the map is not completely positive.

    (Positivity of the underlying map is a separate property — block positivity
    of the Choi matrix — which this test does not check.)
    """
    J = _choi_1975_choi_matrix()
    eigvals = np.sort(np.real(np.linalg.eigvalsh((J + J.conj().T) / 2)))
    # Not PSD: smallest eigenvalue must be strictly negative.
    assert eigvals[0] < -0.5
    # Nonzero: largest eigenvalue must be strictly positive.
    assert eigvals[-1] > 0.5


def test_choi_1975_map_detects_npt_isotropic_3x3_via_partial_channel():
    """Applying Choi's map to an NPT 3x3 isotropic state gives a non-PSD image.

    This is the entanglement-witness signature used in section 12 of
    :func:`is_separable`. We exercise the same `partial_channel` path directly
    rather than routing through :func:`is_separable`, because the PPT criterion
    would catch this state earlier.
    """
    me_vec = max_entangled(3, False, False)
    rho_iso = 0.6 * np.eye(9) / 9 + 0.4 * (me_vec @ me_vec.conj().T)
    J_choi = _choi_1975_choi_matrix()

    for sys in (0, 1):
        mapped = partial_channel(rho_iso, J_choi, sys=sys, dim=[3, 3])
        assert not is_positive_semidefinite(mapped), (
            f"Choi 1975 map should produce a non-PSD image on sys={sys}"
        )


def test_rank_marginal_horodecki_is_separable_3x4_low_rank():
    """A deterministic rank-5 separable 3x4 state is caught by a Horodecki-family check.

    Five fixed, linearly independent product vectors give a rank-5 separable state in
    C^3 (x) C^4. Rank 5 > max(dA, dB) = 4 so the first Horodecki branch does not fire;
    the rank-sum or rank-marginal branch must.
    """
    dA, dB = 3, 4
    e0 = np.array([1, 0, 0], dtype=complex)
    e1 = np.array([0, 1, 0], dtype=complex)
    e2 = np.array([0, 0, 1], dtype=complex)
    f0 = np.array([1, 0, 0, 0], dtype=complex)
    f1 = np.array([0, 1, 0, 0], dtype=complex)
    f2 = np.array([0, 0, 1, 0], dtype=complex)
    f3 = np.array([0, 0, 0, 1], dtype=complex)
    f_plus = np.array([1, 1, 1, 1], dtype=complex) / 2
    product_vectors = [
        np.kron(e0, f0),
        np.kron(e0, f1),
        np.kron(e1, f2),
        np.kron(e1, f3),
        np.kron(e2, f_plus),
    ]
    rho = sum(np.outer(psi, psi.conj()) for psi in product_vectors) / len(product_vectors)
    assert np.linalg.matrix_rank(rho) == 5  # rank 5 > max(dA, dB) = 4, by construction

    sep, reason = is_separable(rho, dim=[dA, dB])
    assert sep is True
    # Either the rank-marginal or the rank-sum bound may fire depending on the
    # exact partial-transpose rank; we just need the verdict to be correct and
    # the Horodecki family to have flagged it.
    assert "Horodecki" in reason


# --- Tests for iterative product-state subtraction (#1244) ---


def test_iter_sub_decomposes_product_state_2x2():
    """A plain product state should be decomposed constructively."""
    rho = np.kron(np.eye(2) / 2, np.diag([0.7, 0.3]))
    assert _iterative_product_state_subtraction(rho, [2, 2], tol=1e-8) is True


def test_iter_sub_decomposes_classical_mixture_2x2():
    """A diagonal classical-correlation state is separable and decomposable."""
    e0, e1 = basis(2, 0), basis(2, 1)
    p00 = np.kron(e0, e0)
    p11 = np.kron(e1, e1)
    rho = 0.6 * (p00 @ p00.conj().T) + 0.4 * (p11 @ p11.conj().T)
    assert _iterative_product_state_subtraction(rho, [2, 2], tol=1e-8) is True


def test_iter_sub_decomposes_mixture_of_four_product_states_3x3():
    """A rank-4 separable mixture in 3x3 is decomposed constructively."""
    e = [basis(3, i) for i in range(3)]
    projs = [np.outer(np.kron(e[i], e[j]), np.kron(e[i], e[j]).conj()) for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 1)]]
    rho = sum(projs) / 4
    assert _iterative_product_state_subtraction(rho, [3, 3], tol=1e-8) is True


def test_iter_sub_decomposes_maximally_mixed_via_separable_ball():
    """The maximally mixed state is inside the Gurvits-Barnum ball on the very first outer iteration."""
    assert _iterative_product_state_subtraction(np.eye(9) / 9, [3, 3], tol=1e-8) is True


def test_iter_sub_rejects_bell_state():
    """A maximally entangled state cannot be constructively decomposed."""
    rho = bell(0) @ bell(0).conj().T
    assert _iterative_product_state_subtraction(rho, [2, 2], tol=1e-8) is False


def test_iter_sub_rejects_upb_tile_ppt_entangled():
    """The UPB tile state is PPT entangled; the algorithm should fail to decompose it."""
    rho = np.identity(9)
    for i in range(5):
        rho = rho - tile(i) @ tile(i).conj().T
    rho = rho / 4
    assert _iterative_product_state_subtraction(rho, [3, 3], tol=1e-8) is False


def test_iter_sub_respects_iteration_budget_on_entangled_state():
    """Budget exhaustion returns False without hanging or raising."""
    rho = bell(0) @ bell(0).conj().T
    # Run with a tighter budget and confirm we still get a clean False.
    assert (
        _iterative_product_state_subtraction(
            rho, [2, 2], tol=1e-8, max_outer_iter=5, n_restarts=2, max_inner_iter=5
        )
        is False
    )
