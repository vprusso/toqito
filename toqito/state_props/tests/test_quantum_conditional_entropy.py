"""Tests for quantum_conditional_entropy."""

import warnings

import cvxpy
import numpy as np
import pytest

from toqito.matrix_ops.partial_trace import partial_trace
from toqito.state_props.petz_renyi_conditional_entropy import (
    petz_renyi_conditional_entropy,
)
from toqito.state_props.quantum_conditional_entropy import quantum_conditional_entropy
from toqito.state_props.quantum_relative_entropy import quantum_relative_entropy
from toqito.state_props.von_neumann_entropy import von_neumann_entropy
from toqito.states import bell

RHO_A = np.array([[0.8, 0.0], [0.0, 0.2]])
RHO_B = np.array([[0.3, 0.0], [0.0, 0.7]])
PRODUCT_STATE = np.kron(RHO_A, RHO_B)
MAX_ENTANGLED_STATE = bell(0) @ bell(0).conj().T
_DIM = [2, 2]

_RHO = np.eye(4) / 4


def _affine_fixed_at(mat: np.ndarray, *, hermitian: bool = True) -> cvxpy.Expression:
    """``Constant(A) + W - W`` with ``W.value = 0`` (algebraically ``A``)."""
    n = mat.shape[0]
    if hermitian:
        w_var = cvxpy.Variable((n, n), hermitian=True)
        w_var.value = np.zeros((n, n), dtype=np.complex128)
    else:
        w_var = cvxpy.Variable((n, n), symmetric=True)
        w_var.value = np.zeros((n, n))
    expr = cvxpy.Constant(mat) + w_var - w_var
    assert expr.is_affine() and not expr.is_constant()
    return expr


def _rand_density(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    mat = g @ g.conj().T + 1e-1 * np.eye(dim)
    mat = (mat + mat.conj().T) / 2
    return mat / np.trace(mat)


def _reference_conditional_entropy(rho: np.ndarray, dim: list[int], sys: int) -> float:
    r"""Build :math:`-D(\rho \| I \otimes \rho_{\text{marginal}})` directly."""
    if sys == 0:
        sigma = np.kron(np.eye(dim[0]), partial_trace(rho, 0, dim))
    else:
        sigma = np.kron(partial_trace(rho, 1, dim), np.eye(dim[1]))
    return -quantum_relative_entropy(rho, sigma)


@pytest.mark.parametrize(
    ("sys", "rho", "dim", "expected_msg"),
    [
        (2, _RHO, _DIM, "sys must be 0 or 1"),
        (-1, _RHO, _DIM, "sys must be 0 or 1"),
        (
            0,
            [[1.0, 0.0], [0.0, 1.0]],
            _DIM,
            "rho must be a numpy array or a cvxpy expression",
        ),
        (0, _RHO, (2, 2), "dim must be a list or numpy array"),
        (0, _RHO, [2], "dim must have length 2"),
        (0, _RHO, [2, 2, 2], "dim must have length 2"),
        (0, _RHO, [2.0, 2], "dim must have integer elements"),
        (0, _RHO, [0, 2], "dim must have positive elements"),
        (0, _RHO, [-1, 2], "dim must have positive elements"),
        (0, _RHO, [3, 3], "dim must match the shape of rho"),
        (0, np.eye(2) / 2, [2, 2], "dim must match the shape of rho"),
    ],
)
def test_quantum_conditional_entropy_invalid_input(
    sys: int,
    rho: object,
    dim: object,
    expected_msg: str,
):
    """Invalid inputs should raise a clear ValueError."""
    with pytest.raises(ValueError, match=expected_msg):
        quantum_conditional_entropy(rho, dim, sys=sys)


def test_product_state_sys_zero_matches_entropy_of_a():
    """For a product state, H(A|B) should equal S(rho_A) in nats."""
    expected = von_neumann_entropy(RHO_A) * np.log(2)
    result = quantum_conditional_entropy(PRODUCT_STATE, _DIM, sys=0)
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-8)


def test_product_state_sys_one_matches_entropy_of_b():
    """For a product state, H(B|A) should equal S(rho_B) in nats."""
    expected = von_neumann_entropy(RHO_B) * np.log(2)
    result = quantum_conditional_entropy(PRODUCT_STATE, _DIM, sys=1)
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-8)


def test_bell_state_sys_zero():
    """For a Bell state, H(A|B) = S(rho_AB) - S(rho_B) = -1 bit = -log(2) nats."""
    expected = -np.log(2)
    result = quantum_conditional_entropy(MAX_ENTANGLED_STATE, _DIM, sys=0)
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-8)


def test_bell_state_sys_one():
    """For a Bell state, H(B|A) = S(rho_AB) - S(rho_A) = -1 bit = -log(2) nats."""
    expected = -np.log(2)
    result = quantum_conditional_entropy(MAX_ENTANGLED_STATE, _DIM, sys=1)
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("sys", [0, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_negative_relative_entropy_formula(sys: int, seed: int):
    """Should agree with -D(rho || I ⊗ rho_marginal) built from partial trace."""
    rho = _rand_density(4, seed)
    expected = _reference_conditional_entropy(rho, _DIM, sys)
    result = quantum_conditional_entropy(rho, _DIM, sys=sys)
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-8)


def test_agrees_with_petz_alpha_one_in_nats():
    """Petz downarrow at alpha=1 is H(A|B) in log2; multiply by ln(2) for nats."""
    for rho in (PRODUCT_STATE, MAX_ENTANGLED_STATE):
        petz = petz_renyi_conditional_entropy(rho, 1.0, dim=_DIM, variant="downarrow")
        expected = petz * np.log(2)
        result = quantum_conditional_entropy(rho, _DIM, sys=0)
        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)


def test_von_neumann_difference_in_nats_sys_one():
    """H(B|A) should equal (S(rho) - S(rho_A)) converted from log2 to nats."""
    for rho in (PRODUCT_STATE, MAX_ENTANGLED_STATE):
        rho_a = partial_trace(rho, 1, _DIM)
        expected = (von_neumann_entropy(rho) - von_neumann_entropy(rho_a)) * np.log(2)
        result = quantum_conditional_entropy(rho, _DIM, sys=1)
        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)


def test_affine_expression_matches_numeric_evaluation():
    """Affine CVXPY rho should evaluate to the same value as the numeric matrix."""
    rho = PRODUCT_STATE
    numeric = quantum_conditional_entropy(rho, _DIM, sys=0)
    affine_val = quantum_conditional_entropy(_affine_fixed_at(rho), _DIM, sys=0)
    np.testing.assert_allclose(float(affine_val), numeric, rtol=1e-4, atol=1e-4)


@pytest.mark.slow
def test_sdp_maximize_over_density_matrices():
    """Maximizing conditional entropy over 2-qubit states should solve as an SDP."""
    rho_var = cvxpy.Variable((4, 4), hermitian=True)
    rho_var.value = PRODUCT_STATE
    obj = quantum_conditional_entropy(rho_var, _DIM, sys=0)
    prob = cvxpy.Problem(
        cvxpy.Maximize(obj),
        [rho_var >> 0, cvxpy.trace(rho_var) == 1],
    )
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val is not None


def test_pure_state_does_not_leak_singular_logm_warning():
    """A rank-deficient input must not surface scipy's singular-logm warning.

    ``quantum_conditional_entropy`` routes through ``quantum_relative_entropy``'s numeric ``logm``
    path, which is evaluated on singular operators for pure and reduced states. The result is valid,
    so the cosmetic ``"The logm input matrix is exactly singular."`` warning must be suppressed at
    the source rather than by callers.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = quantum_conditional_entropy(MAX_ENTANGLED_STATE, _DIM, sys=0)
    assert np.isclose(result, -np.log(2))
    assert not any("logm input matrix is exactly singular" in str(w.message) for w in record)
