"""Unit tests for bell_inequality_max function."""
import warnings
from unittest.mock import PropertyMock

import cvxpy
import numpy as np
import pytest

from toqito.helper.bell_notation_conversions import fc2cg, fc2fp, fp2cg
from toqito.nonlocal_games.bell_inequality_max import _integer_digits, bell_inequality_max

# Define CHSH coefficients in different notations
# CHSH = A0B0 + A0B1 + A1B0 - A1B1
# FC: [[<1>, <B0>, <B1>], [<A0>, <A0B0>, <A0B1>], [<A1>, <A1B0>, <A1B1>]]
CHSH_FC = np.array([[0, 0, 0],
                    [0, 1, 1],
                    [0, 1, -1]])
# CG: [[<1>, pB(0|0), pB(0|1)], [pA(0|0), p(00|00), p(00|01)], [pA(0|1), p(00|10), p(00|11)]]
CHSH_CG = fc2cg(CHSH_FC, behaviour=False)
# FP: P(a,b|x,y) coefficients (order a,b,x,y)
CHSH_FP = fc2fp(CHSH_FC, behaviour=False)

# Bell scenario description [oa, ob, ma, mb]
DESC_CHSH = [2, 2, 2, 2]

# Expected values for CHSH
CHSH_CLASSICAL_MAX = 2.0
CHSH_QUANTUM_MAX = 2 * np.sqrt(2)
CHSH_NOSIGNAL_MAX = 4.0

# Define CGLMP(d=3) coefficients
def _get_cglmp_fp_coeffs(dim):
    coeffs = np.zeros((dim, dim, 2, 2))
    for k in range(dim // 2):
        factor = (1 - 2 * k / (dim - 1))
        for a in range(dim):
            for b in range(dim):
                # x=0, y=0
                if a == (b + k) % dim:
                    coeffs[a, b, 0, 0] += factor
                if a == (b - k - 1) % dim:
                    coeffs[a, b, 0, 0] -= factor
                # x=0, y=1
                if b == (a + k) % dim:
                    coeffs[a, b, 0, 1] += factor
                if b == (a - k - 1) % dim:
                    coeffs[a, b, 0, 1] -= factor
                # x=1, y=0
                if b == (a + k + 1) % dim:
                    coeffs[a, b, 1, 0] += factor
                if b == (a - k) % dim:
                    coeffs[a, b, 1, 0] -= factor
                 # x=1, y=1
                if a == (b + k) % dim:
                    coeffs[a, b, 1, 1] += factor
                if a == (b - k - 1) % dim:
                    coeffs[a, b, 1, 1] -= factor
    return coeffs

CGLMP3_FP = _get_cglmp_fp_coeffs(3)
# We still generate CGLMP3_CG for testing quantum/NS cases, assuming fp2cg is correct.
CGLMP3_CG = fp2cg(CGLMP3_FP, behaviour=False)
DESC_CGLMP3 = [3, 3, 2, 2]
CGLMP3_QUANTUM_K2_MAX = 2.914
CGLMP3_CLASSICAL_MAX = 2.0


# --- Test Helper Functions ---

def test_integer_digits_warning():
    """Test warning when number is too large for digits."""
    with pytest.warns(UserWarning, match="too large for base 2 and 3 digits"):
        _integer_digits(10, 2, 3)
    # Also test the case where it fits exactly (should not warn)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _integer_digits(7, 2, 3)


# --- Test Classical Bounds ---

@pytest.mark.parametrize("coeffs, notation", [(CHSH_FC, 'fc'), (CHSH_CG, 'cg'), (CHSH_FP, 'fp')])
def test_chsh_classical(coeffs, notation):
    """Test CHSH classical maximum."""
    bmax = bell_inequality_max(coeffs, DESC_CHSH, notation, mtype='classical')
    np.testing.assert_almost_equal(bmax, CHSH_CLASSICAL_MAX, decimal=5)


@pytest.mark.parametrize("coeffs, notation", [(CGLMP3_FP, 'fp')])
def test_cglmp3_classical(coeffs, notation):
    """Test CGLMP(d=3) classical maximum (only FP input tested)."""
    bmax = bell_inequality_max(coeffs, DESC_CGLMP3, notation, mtype='classical')
    np.testing.assert_almost_equal(bmax, CGLMP3_CLASSICAL_MAX, decimal=5)


def test_cglmp3_classical_cg_unsupported():
    """Test that CG input raises NotImplementedError for general classical case."""
    with pytest.raises(NotImplementedError):
        bell_inequality_max(CGLMP3_CG, DESC_CGLMP3, 'cg', mtype='classical')


# --- Test Quantum Bounds (NPA) ---

@pytest.mark.cvxpy
@pytest.mark.parametrize("coeffs, desc, notation, k_level, expected", [
    (CHSH_FC, DESC_CHSH, 'fc', 1, CHSH_QUANTUM_MAX),
    (CHSH_CG, DESC_CHSH, 'cg', 1, CHSH_QUANTUM_MAX),
    (CHSH_FP, DESC_CHSH, 'fp', 1, CHSH_QUANTUM_MAX),
    (CHSH_FC, DESC_CHSH, 'fc', '1+ab', CHSH_QUANTUM_MAX),
    (CHSH_CG, DESC_CHSH, 'cg', '1+ab', CHSH_QUANTUM_MAX),
    (CHSH_FP, DESC_CHSH, 'fp', '1+ab', CHSH_QUANTUM_MAX),
    (CGLMP3_FP, DESC_CGLMP3, 'fp', 2, CGLMP3_QUANTUM_K2_MAX),
    (CGLMP3_CG, DESC_CGLMP3, 'cg', 2, CGLMP3_QUANTUM_K2_MAX),
    (CGLMP3_FP, DESC_CGLMP3, 'fp', '1+ab+aab+baa', CGLMP3_QUANTUM_K2_MAX),
    (CGLMP3_CG, DESC_CGLMP3, 'cg', '1+ab+aab+baa', CGLMP3_QUANTUM_K2_MAX),
])
def test_quantum_bounds(coeffs, desc, notation, k_level, expected):
    """Test quantum (NPA) upper bounds."""
    solver_to_use = cvxpy.SCS if cvxpy.SCS in cvxpy.installed_solvers() else None
    if solver_to_use is None and cvxpy.MOSEK in cvxpy.installed_solvers():
         solver_to_use = cvxpy.MOSEK

    if solver_to_use is None:
        pytest.skip("Requires an SDP solver like SCS or MOSEK")

    bmax = bell_inequality_max(coeffs, desc,
                               notation, mtype='quantum', k=k_level,
                               solver=solver_to_use)
    np.testing.assert_almost_equal(bmax, expected, decimal=3)


# --- Test No-Signaling Bounds ---

@pytest.mark.cvxpy
@pytest.mark.parametrize("coeffs, desc, notation, expected", [
    (CHSH_FC, DESC_CHSH, 'fc', CHSH_NOSIGNAL_MAX),
    (CHSH_CG, DESC_CHSH, 'cg', CHSH_NOSIGNAL_MAX),
    (CHSH_FP, DESC_CHSH, 'fp', CHSH_NOSIGNAL_MAX),
    (CGLMP3_FP, DESC_CGLMP3, 'fp', 4.0),
    (CGLMP3_CG, DESC_CGLMP3, 'cg', 4.0),
])
def test_nosignal(coeffs, desc, notation, expected):
    """Test no-signaling maximum."""
    solver_to_use = cvxpy.SCS if cvxpy.SCS in cvxpy.installed_solvers() else None
    if solver_to_use is None and cvxpy.MOSEK in cvxpy.installed_solvers():
         solver_to_use = cvxpy.MOSEK
    if solver_to_use is None and cvxpy.ECOS in cvxpy.installed_solvers():
        solver_to_use = cvxpy.ECOS

    if solver_to_use is None:
        pytest.skip("Requires an LP solver like SCS, MOSEK, or ECOS")

    bmax = bell_inequality_max(coeffs, desc, notation, mtype='nosignal', solver=solver_to_use)
    np.testing.assert_almost_equal(bmax, expected, decimal=4)


# --- Test Input Validation ---

def test_invalid_notation():
    """Test error on invalid notation."""
    with pytest.raises(ValueError, match="Invalid notation"):
        bell_inequality_max(CHSH_FC, DESC_CHSH, 'invalid')

def test_invalid_mtype():
    """Test error on invalid mtype."""
    with pytest.raises(ValueError, match="Invalid mtype"):
        bell_inequality_max(CHSH_FC, DESC_CHSH, 'fc', mtype='invalid')

def test_invalid_desc_length():
    """Test error on invalid desc length."""
    with pytest.raises(ValueError, match="desc must be a list of length 4"):
        bell_inequality_max(CHSH_FC, [2, 2, 2], 'fc')

def test_fc_non_binary_outputs():
    """Test error for FC notation with non-binary outputs."""
    ma, mb = 2, 2
    coeffs_dummy_fc = np.zeros((ma + 1, mb + 1))
    with pytest.raises(ValueError, match="notation requires binary outcomes"):
        bell_inequality_max(coeffs_dummy_fc, [3, 2, ma, mb], 'fc')

def test_fc_shape_mismatch():
    """Test error for FC notation with wrong coefficient shape."""
    desc = [2, 2, 2, 2] # ma=2, mb=2, expects (3, 3) shape
    wrong_coeffs_fc = np.zeros((2, 3)) # Incorrect shape
    with pytest.raises(ValueError, match="FC coefficients shape mismatch"):
        bell_inequality_max(wrong_coeffs_fc, desc, 'fc')

def test_cg_shape_mismatch():
    """Test error for CG notation with wrong coefficient shape."""
    with pytest.raises(ValueError, match="CG coefficients shape mismatch"):
        bell_inequality_max(np.zeros((5, 4)), [3, 2, 2, 2], 'cg')

def test_fp_shape_mismatch():
    """Test error for FP notation with wrong coefficient shape."""
    with pytest.raises(ValueError, match="FP coefficients shape mismatch"):
        bell_inequality_max(np.zeros((2, 2, 3, 2)), [2, 2, 2, 2], 'fp')


# --- Test Classical Strategy Swapping ---

def test_classical_swap():
    """Test classical calculation when ma < mb."""
    desc_swap = [2, 2, 2, 3]
    coeffs_swap_fc = np.zeros((3, 4))
    coeffs_swap_fc[1, 1] = 1.0
    expected_max = 1.0
    bmax = bell_inequality_max(coeffs_swap_fc, desc_swap, 'fc', mtype='classical')
    np.testing.assert_almost_equal(bmax, expected_max)

    desc_gen_swap = [3, 2, 2, 3]
    coeffs_gen_swap_fp = np.zeros((3, 2, 2, 3))
    coeffs_gen_swap_fp[0, 0, 0, 0] = 1.0
    expected_gen_max = 1.0
    bmax_gen = bell_inequality_max(coeffs_gen_swap_fp, desc_gen_swap, 'fp', mtype='classical')
    np.testing.assert_almost_equal(bmax_gen, expected_gen_max)


def test_classical_general_swap_triggered():
    """Test classical general calculation triggers party swapping."""
    desc_swap_gen = [2, 3, 3, 2]
    oa, ob, ma, mb = desc_swap_gen
    coeffs_fp_swap = np.zeros((oa, ob, ma, mb))
    coeffs_fp_swap[0, 0, 0, 0] = 1.0
    expected_max = 1.0
    bmax = bell_inequality_max(coeffs_fp_swap, desc_swap_gen, 'fp', mtype='classical')
    np.testing.assert_almost_equal(bmax, expected_max)


# --- Test CVXPY Solver Status Handling ---

@pytest.mark.cvxpy
@pytest.mark.parametrize("status_to_mock, expected_message", [
    (cvxpy.INFEASIBLE, "Problem is infeasible"),
    (cvxpy.UNBOUNDED, "Problem is unbounded"),
    (cvxpy.SOLVER_ERROR, "Optimization failed due to solver issue or limits. Status: solver_error"),
    (cvxpy.INFEASIBLE_INACCURATE, "Problem is infeasible"),
    (cvxpy.UNBOUNDED_INACCURATE, "Problem is unbounded"),
])
def test_quantum_solver_failure_statuses(mocker, status_to_mock, expected_message):
    """Test handling of CVXPY failure statuses."""
    coeffs = CHSH_FP
    desc = DESC_CHSH
    notation = 'fp'
    mtype = 'quantum'

    _ = mocker.patch('cvxpy.Problem.solve', return_value=-1.0)
    mocker.patch(
        'cvxpy.Problem.status',
        new_callable=PropertyMock,
        return_value=status_to_mock
    )

    with pytest.raises(RuntimeError, match=expected_message):
        bell_inequality_max(coeffs, desc, notation, mtype=mtype, k=1)


@pytest.mark.cvxpy
def test_quantum_solver_inaccurate(mocker):
    """Test warning for OPTIMAL_INACCURATE status."""
    coeffs = CHSH_FP
    desc = DESC_CHSH
    notation = 'fp'
    mtype = 'quantum'

    _ = mocker.patch('cvxpy.Problem.solve', return_value=2.8)
    mocker.patch(
        'cvxpy.Problem.status',
        new_callable=PropertyMock,
        return_value=cvxpy.OPTIMAL_INACCURATE
    )

    with pytest.warns(RuntimeWarning, match="Solver finished with status: optimal_inaccurate"):
         result = bell_inequality_max(coeffs, desc, notation, mtype=mtype, k=1)
         assert abs(result - 2.8) < 1e-6


@pytest.mark.cvxpy
def test_quantum_solver_returns_none(mocker):
    """Test error when solver returns None."""
    coeffs = CHSH_FP
    desc = DESC_CHSH
    notation = 'fp'
    mtype = 'quantum'

    _ = mocker.patch('cvxpy.Problem.solve', return_value=None)
    mocker.patch(
        'cvxpy.Problem.status',
        new_callable=PropertyMock,
        return_value=cvxpy.OPTIMAL
    )

    with pytest.raises(RuntimeError, match="solver failed to return a value"):
        bell_inequality_max(coeffs, desc, notation, mtype=mtype, k=1)
