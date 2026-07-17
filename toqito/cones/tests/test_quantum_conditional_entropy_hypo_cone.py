"""Tests for quantum_conditional_entropy_hypo_cone."""

import re

import cvxpy
import numpy as np
import pytest

from toqito.cones.quantum_conditional_entropy_hypo_cone import (
    quantum_conditional_entropy_hypo_cone,
)
from toqito.state_props.quantum_conditional_entropy import quantum_conditional_entropy
from toqito.states import bell

_DIM = [2, 2]
RHO_A = np.array([[0.8, 0.0], [0.0, 0.2]])
RHO_B = np.array([[0.3, 0.0], [0.0, 0.7]])
PRODUCT_STATE = np.kron(RHO_A, RHO_B)
MAX_ENTANGLED_STATE = bell(0) @ bell(0).conj().T
_RHO = np.eye(4) / 4


def _rand_density(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((dim, dim))
    mat = g @ g.T + 1e-1 * np.eye(dim)
    mat = (mat + mat.T) / 2
    return mat / np.trace(mat)


@pytest.mark.parametrize("sys", [0, 1])
@pytest.mark.parametrize(
    "rho",
    [PRODUCT_STATE, MAX_ENTANGLED_STATE],
    ids=["product", "bell"],
)
def test_quantum_conditional_entropy_hypo_cone_bell_and_product(sys: int, rho: np.ndarray):
    """Maximize ``t`` at fixed Bell / product Constants vs float."""
    ref = quantum_conditional_entropy(rho, _DIM, sys=sys)
    t = cvxpy.Variable()
    cons = quantum_conditional_entropy_hypo_cone(cvxpy.Constant(rho), t, _DIM, sys=sys)
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(ref, abs=2e-2)


def test_quantum_conditional_entropy_hypo_cone_hermitian():
    """``hermitian=True`` takes the real part of the QRE epigraph bound."""
    rho = MAX_ENTANGLED_STATE.astype(complex)
    ref = quantum_conditional_entropy(rho, _DIM, sys=0)
    t = cvxpy.Variable()
    cons = quantum_conditional_entropy_hypo_cone(cvxpy.Constant(rho), t, _DIM, sys=0, hermitian=True)
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(ref, abs=2e-2)


@pytest.mark.parametrize("sys", [0, 1])
@pytest.mark.parametrize("seed", [0, 1])
def test_quantum_conditional_entropy_hypo_cone_at_constant(sys: int, seed: int):
    """Random density Constants match numeric conditional entropy."""
    rho = _rand_density(4, seed + 10 * sys)
    ref = quantum_conditional_entropy(rho, _DIM, sys=sys)
    t = cvxpy.Variable()
    cons = quantum_conditional_entropy_hypo_cone(cvxpy.Constant(rho), t, _DIM, sys=sys)
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(ref, abs=3e-2)


def test_quantum_conditional_entropy_hypo_cone_composition():
    """Free density; maximize ``H(A|B)`` recovers the maximally mixed state."""
    n = 4
    rho_var = cvxpy.Variable((n, n), symmetric=True)
    t = cvxpy.Variable()
    cons = quantum_conditional_entropy_hypo_cone(rho_var, t, _DIM, sys=0, hermitian=False)
    cons.extend([rho_var >> 0, cvxpy.trace(rho_var) == 1])
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)

    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(float(np.log(2)), abs=5e-2)
    assert rho_var.value is not None
    np.testing.assert_allclose(rho_var.value, np.eye(n) / n, atol=5e-2)


@pytest.mark.parametrize(
    ("sys", "rho", "dim", "expected_msg"),
    [
        (2, _RHO, _DIM, "sys must be 0 or 1"),
        (-1, _RHO, _DIM, "sys must be 0 or 1"),
        (0, _RHO, (2, 2), "dim must be a list or numpy array"),
        (0, _RHO, [2], "dim must have length 2"),
        (0, _RHO, [2.0, 2], "dim must have integer elements"),
        (0, _RHO, [0, 2], "dim must have positive elements"),
        (0, _RHO, [3, 3], "dim must match the shape of rho"),
    ],
)
def test_quantum_conditional_entropy_hypo_cone_invalid_input(
    sys: int,
    rho: np.ndarray,
    dim: object,
    expected_msg: str,
):
    """Invalid ``sys`` / ``dim`` raise clear ``ValueError``s."""
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        quantum_conditional_entropy_hypo_cone(cvxpy.Constant(rho), t, dim, sys=sys)


def test_quantum_conditional_entropy_hypo_cone_rho_not_square() -> None:
    """Reject non-square ``rho``."""
    rho = cvxpy.Variable((2, 3))
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("rho must be square.")):
        quantum_conditional_entropy_hypo_cone(rho, t, [2, 2], sys=0)
