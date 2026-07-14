"""Tests for ln_quantum_entropy_hypo_cone."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import re

import cvxpy
import numpy as np
import pytest
from scipy.linalg import logm

from toqito.cones.ln_quantum_entropy_hypo_cone import ln_quantum_entropy_hypo_cone
from toqito.matrix_props import is_positive_semidefinite
from toqito.state_props.ln_quantum_entropy import ln_quantum_entropy


def _rand_psd_normalized(dim: int, seed: int, *, hermitian: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if hermitian:
        g = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    else:
        g = rng.standard_normal((dim, dim))
    mat = g @ g.conj().T + 1e-1 * np.eye(dim, dtype=g.dtype)
    mat = (mat + mat.conj().T) / 2
    return mat / np.trace(mat)


def _entropy_reference(mat_x: np.ndarray) -> float:
    return float(np.real(-np.trace(mat_x @ logm(mat_x))))


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mk", [1, 3])
@pytest.mark.parametrize("apx", [-1, 0, 1])
@pytest.mark.parametrize("hermitian", [False, True])
def test_ln_quantum_entropy_hypo_cone_at_constant(dim: int, mk: int, apx: int, hermitian: bool):
    """Maximize ``t`` at fixed ``Constant(X)`` and compare to exact entropy."""
    if mk == 1 and apx == 0:
        pytest.skip("CVXQUAD skips (m,k)=(1,1) with Pade apx=0.")

    seed = dim * 100_003 + mk * 17 + (apx + 1) * 3 + int(hermitian)
    mat_x = _rand_psd_normalized(dim, seed, hermitian=hermitian)
    assert is_positive_semidefinite(np.asarray(mat_x, dtype=np.complex128))

    h_ref = _entropy_reference(mat_x)
    np.testing.assert_allclose(ln_quantum_entropy(mat_x), h_ref, rtol=1e-8, atol=1e-8)

    t = cvxpy.Variable()
    cons = ln_quantum_entropy_hypo_cone(
        cvxpy.Constant(mat_x),
        t,
        m=mk,
        k=mk,
        apx=apx,
        hermitian=hermitian,
    )
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    cvx_val = prob.solve(solver=cvxpy.SCS, verbose=False)

    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert cvx_val is not None

    if abs(h_ref) < 1e-12:
        assert abs(cvx_val - h_ref) <= 1e-4
        return

    relerr = (cvx_val - h_ref) / abs(h_ref)
    assert apx * relerr >= -1e-4, relerr
    if mk >= 3:
        assert abs(relerr) <= 1e-2, relerr


def test_ln_quantum_entropy_hypo_cone_composition():
    """Free ``Variable`` matrix with ``tr(X)=1``, ``X >> 0``; maximize entropy via the hypo cone."""
    n = 2
    x_var = cvxpy.Variable((n, n), symmetric=True)
    t = cvxpy.Variable()
    cons = ln_quantum_entropy_hypo_cone(x_var, t, m=3, k=3, apx=0, hermitian=False)
    cons.extend([x_var >> 0, cvxpy.trace(x_var) == 1])
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)

    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    # Maximally mixed state maximizes von Neumann entropy among density matrices.
    assert val == pytest.approx(float(np.log(n)), rel=0, abs=5e-2)
    assert x_var.value is not None
    np.testing.assert_allclose(x_var.value, np.eye(n) / n, atol=5e-2)


def test_ln_quantum_entropy_hypo_cone_maximally_mixed_constant():
    """Sanity check: ``Maximize(t)`` at ``I/d`` recovers ``log(d)``."""
    n = 2
    mat_x = np.eye(n) / n
    t = cvxpy.Variable()
    cons = ln_quantum_entropy_hypo_cone(cvxpy.Constant(mat_x), t, m=3, k=3, apx=0)
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(float(np.log(n)), rel=0, abs=1e-2)


def test_ln_quantum_entropy_hypo_cone_mat_x_not_square() -> None:
    """Reject non-square ``mat_x``."""
    mat_x = cvxpy.Variable((2, 3))
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("mat_x must be square.")):
        ln_quantum_entropy_hypo_cone(mat_x, t)


def test_ln_quantum_entropy_hypo_cone_m_invalid() -> None:
    """Reject ``m`` below 1."""
    mat_x = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("m must be at least 1")):
        ln_quantum_entropy_hypo_cone(mat_x, t, m=0)


def test_ln_quantum_entropy_hypo_cone_k_invalid() -> None:
    """Reject ``k`` below 1."""
    mat_x = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("k must be at least 1")):
        ln_quantum_entropy_hypo_cone(mat_x, t, k=0)


def test_ln_quantum_entropy_hypo_cone_apx_invalid() -> None:
    """Reject ``apx`` outside ``{-1, 0, 1}``."""
    mat_x = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("apx must be -1, 0, or 1")):
        ln_quantum_entropy_hypo_cone(mat_x, t, apx=2)
