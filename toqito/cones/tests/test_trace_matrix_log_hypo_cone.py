"""Tests for trace_matrix_log_hypo_cone."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import re

import cvxpy
import numpy as np
import pytest
from scipy.linalg import logm

from toqito.cones.trace_matrix_log_hypo_cone import trace_matrix_log_hypo_cone
from toqito.matrix_props import is_positive_semidefinite
from toqito.matrix_props.trace_matrix_log import trace_matrix_log


def _rand_psd(dim: int, seed: int, *, hermitian: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if hermitian:
        g = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    else:
        g = rng.standard_normal((dim, dim))
    mat = g @ g.conj().T + 1e-1 * np.eye(dim, dtype=g.dtype)
    return (mat + mat.conj().T) / 2


def _rand_psd_normalized(dim: int, seed: int, *, hermitian: bool) -> np.ndarray:
    mat = _rand_psd(dim, seed, hermitian=hermitian)
    return mat / np.trace(mat)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mk", [1, 3])
@pytest.mark.parametrize("apx", [-1, 0, 1])
@pytest.mark.parametrize("hermitian", [False, True])
def test_trace_matrix_log_hypo_cone_at_constant(dim: int, mk: int, apx: int, hermitian: bool):
    """Maximize ``t`` at fixed ``Constant(A)`` and compare to ``tr(C log A)``."""
    if mk == 1 and apx == 0:
        pytest.skip("CVXQUAD skips (m,k)=(1,1) with Pade apx=0.")

    seed = dim * 100_003 + mk * 17 + (apx + 1) * 3 + int(hermitian)
    mat_a = _rand_psd_normalized(dim, seed, hermitian=hermitian)
    mat_c = _rand_psd(dim, seed + 1, hermitian=hermitian)
    assert is_positive_semidefinite(np.asarray(mat_a, dtype=np.complex128))
    assert is_positive_semidefinite(np.asarray(mat_c, dtype=np.complex128))

    tr_ref = float(np.real(np.trace(mat_c @ logm(mat_a))))
    np.testing.assert_allclose(trace_matrix_log(mat_a, mat_c), tr_ref, rtol=1e-8, atol=1e-8)

    t = cvxpy.Variable()
    cons = trace_matrix_log_hypo_cone(
        cvxpy.Constant(mat_a),
        t,
        mat_c,
        m=mk,
        k=mk,
        apx=apx,
        hermitian=hermitian,
    )
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    cvx_val = prob.solve(solver=cvxpy.SCS, verbose=False)

    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert cvx_val is not None

    if abs(tr_ref) < 1e-12:
        assert abs(cvx_val - tr_ref) <= 1e-4
        return

    relerr = (cvx_val - tr_ref) / abs(tr_ref)
    # SCS can slightly overshoot the analytic bound on larger dims.
    assert apx * relerr >= -1e-3, relerr
    if mk >= 3:
        assert abs(relerr) <= 1e-2, relerr


def test_trace_matrix_log_hypo_cone_default_mat_c_is_identity():
    """Omitted ``mat_c`` uses the identity weight."""
    n = 2
    mat_x = np.eye(n) / n
    t = cvxpy.Variable()
    cons = trace_matrix_log_hypo_cone(cvxpy.Constant(mat_x), t, m=3, k=3, apx=0)
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    ref = float(np.real(np.trace(logm(mat_x))))
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(ref, rel=0, abs=1e-2)


def test_trace_matrix_log_hypo_cone_composition():
    """Free ``Variable`` with ``X >> eps I``; maximize ``tr(log X)`` via the hypo cone."""
    n = 2
    eps = 0.2
    x_var = cvxpy.Variable((n, n), symmetric=True)
    t = cvxpy.Variable()
    cons = trace_matrix_log_hypo_cone(x_var, t, m=3, k=3, apx=0, hermitian=False)
    # Bounded feasible set so Maximize(tr(log X)) is well-posed.
    cons.extend([x_var >> eps * np.eye(n), x_var << np.eye(n)])
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)

    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    # Optimum of tr(log X) on eps I ⪯ X ⪯ I is at X = I.
    assert val == pytest.approx(0.0, abs=5e-2)
    assert x_var.value is not None
    np.testing.assert_allclose(x_var.value, np.eye(n), atol=5e-2)


def test_trace_matrix_log_hypo_cone_mat_x_not_square() -> None:
    """Reject non-square ``mat_x``."""
    mat_x = cvxpy.Variable((2, 3))
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("mat_x must be square.")):
        trace_matrix_log_hypo_cone(mat_x, t)


def test_trace_matrix_log_hypo_cone_mat_c_wrong_type() -> None:
    """Reject non-numpy ``mat_c``."""
    mat_x = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("mat_c must be a numpy array")):
        trace_matrix_log_hypo_cone(mat_x, t, mat_c=cvxpy.Constant(np.eye(2)))


def test_trace_matrix_log_hypo_cone_mat_c_not_psd() -> None:
    """Reject non-PSD ``mat_c``."""
    mat_x = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(
        ValueError,
        match=re.escape("mat_c must be a positive semidefinite matrix"),
    ):
        trace_matrix_log_hypo_cone(mat_x, t, mat_c=np.diag([1.0, -0.5]))


def test_trace_matrix_log_hypo_cone_shape_mismatch() -> None:
    """Reject mismatched ``mat_x`` and ``mat_c`` shapes."""
    mat_x = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(
        ValueError,
        match=re.escape("mat_x and mat_c must have the same shape"),
    ):
        trace_matrix_log_hypo_cone(mat_x, t, mat_c=np.eye(3))


def test_trace_matrix_log_hypo_cone_m_invalid() -> None:
    """Reject ``m`` below 1."""
    mat_x = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("m must be at least 1")):
        trace_matrix_log_hypo_cone(mat_x, t, m=0)


def test_trace_matrix_log_hypo_cone_k_invalid() -> None:
    """Reject ``k`` below 1."""
    mat_x = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("k must be at least 1")):
        trace_matrix_log_hypo_cone(mat_x, t, k=0)


def test_trace_matrix_log_hypo_cone_apx_invalid() -> None:
    """Reject ``apx`` outside ``{-1, 0, 1}``."""
    mat_x = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("apx must be -1, 0, or 1")):
        trace_matrix_log_hypo_cone(mat_x, t, apx=2)
