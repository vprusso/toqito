"""Tests for tsallis_entropy_hypo_cone."""

import re
from fractions import Fraction

import cvxpy
import numpy as np
import pytest

from toqito.cones.tsallis_entropy_hypo_cone import tsallis_entropy_hypo_cone
from toqito.matrix_props import is_positive_semidefinite
from toqito.state_props.tsallis_entropy import tsallis_entropy

DIMS = (2, 3)
ORDERS = (0.5, 0.25, 1.0)
PD_SHIFT = 1e-1


def _case_seed(dim: int, order: float, *, hermitian: bool) -> int:
    r = Fraction(float(order)).limit_denominator()
    return int(dim * 1_000_003 + r.numerator * 10_009 + r.denominator * 100 + int(hermitian))


def _random_pd_normalized(dim: int, seed: int, *, hermitian: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if hermitian:
        x_mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    else:
        x_mat = rng.standard_normal((dim, dim))
    mat = x_mat @ x_mat.conj().T + PD_SHIFT * np.eye(dim, dtype=x_mat.dtype)
    mat = (mat + mat.conj().T) / 2
    return mat / np.trace(mat)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("hermitian", [False, True])
def test_tsallis_entropy_hypo_cone_at_constant(dim: int, order: float, hermitian: bool):
    """Maximize ``t`` at fixed ``Constant(X)`` and compare to numeric Tsallis entropy."""
    seed = _case_seed(dim, order, hermitian=hermitian)
    mat_x = _random_pd_normalized(dim, seed, hermitian=hermitian)
    assert is_positive_semidefinite(np.asarray(mat_x, dtype=np.complex128))

    ref = tsallis_entropy(mat_x, order)

    t = cvxpy.Variable()
    cons = tsallis_entropy_hypo_cone(
        cvxpy.Constant(mat_x),
        t,
        order,
        hermitian=hermitian,
    )
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    cvx_val = prob.solve(solver=cvxpy.SCS, verbose=False)

    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert cvx_val is not None
    np.testing.assert_allclose(float(cvx_val), ref, rtol=2e-2, atol=1e-2)


def test_tsallis_entropy_hypo_cone_maximally_mixed():
    """Sanity check: qubit maximally mixed state at order ``1/2``."""
    mat_x = np.eye(2) / 2
    ref = tsallis_entropy(mat_x, 0.5)
    t = cvxpy.Variable()
    cons = tsallis_entropy_hypo_cone(cvxpy.Constant(mat_x), t, 0.5)
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(ref, abs=2e-2)


def test_tsallis_entropy_hypo_cone_composition():
    """Free density matrix; maximize Tsallis entropy recovers the maximally mixed state."""
    n = 2
    order = 0.5
    x_var = cvxpy.Variable((n, n), symmetric=True)
    t = cvxpy.Variable()
    cons = tsallis_entropy_hypo_cone(x_var, t, order, hermitian=False)
    cons.extend([x_var >> 0, cvxpy.trace(x_var) == 1])
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)

    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    ref = tsallis_entropy(np.eye(n) / n, order)
    assert val == pytest.approx(ref, abs=5e-2)
    assert x_var.value is not None
    np.testing.assert_allclose(x_var.value, np.eye(n) / n, atol=5e-2)


def test_tsallis_entropy_hypo_cone_mat_x_not_square() -> None:
    """Reject non-square ``mat_x``."""
    mat_x = cvxpy.Variable((2, 3))
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("mat_x must be square.")):
        tsallis_entropy_hypo_cone(mat_x, t, 0.5)


def test_tsallis_entropy_hypo_cone_order_zero():
    """``order == 0`` delegates to von Neumann entropy hypograph."""
    mat_x = np.diag([0.25, 0.75])
    ref = tsallis_entropy(mat_x, 0.0)
    t = cvxpy.Variable()
    cons = tsallis_entropy_hypo_cone(cvxpy.Constant(mat_x), t, 0.0)
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(ref, abs=2e-2)


def test_tsallis_entropy_hypo_cone_order_invalid() -> None:
    """Reject ``order`` outside ``[0, 1]``."""
    mat_x = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("order must be in the range [0, 1]")):
        tsallis_entropy_hypo_cone(mat_x, t, 1.5)
    with pytest.raises(ValueError, match=re.escape("order must be in the range [0, 1]")):
        tsallis_entropy_hypo_cone(mat_x, t, -0.1)
