"""Tests for trace_matrix_log."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import re

import cvxpy
import numpy as np
import pytest
from scipy.linalg import logm

from toqito.cones._utils import _AFFINE_VARIABLE_USE_CONE
from toqito.cones.trace_matrix_log_hypo_cone import trace_matrix_log_hypo_cone
from toqito.matrix_props import is_positive_semidefinite
from toqito.matrix_props.trace_matrix_log import trace_matrix_log

_NOT_SUPPORTED = re.escape(_AFFINE_VARIABLE_USE_CONE)


def _rand_psd(dim: int, seed: int, *, hermitian: bool) -> np.ndarray:
    """Random PSD matrix (not necessarily trace-normalized)."""
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


def _trace_matrix_log_sdp_at_fixed_a(
    mat_a: np.ndarray,
    mat_c: np.ndarray,
    *,
    m: int,
    k: int,
    apx: int,
) -> tuple[float, str]:
    """SDP for ``trace_matrix_log`` at fixed ``A = mat_a`` via the hypo cone."""
    is_cplx = bool(np.any(np.imag(mat_a) != 0) or np.any(np.imag(mat_c) != 0))
    t = cvxpy.Variable()
    cons = trace_matrix_log_hypo_cone(
        cvxpy.Constant(mat_a),
        t,
        mat_c,
        m=m,
        k=k,
        apx=apx,
        hermitian=is_cplx,
    )
    prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    return val, prob.status


@pytest.mark.parametrize("dim", [3, 5, 8])
@pytest.mark.parametrize("mk", [1, 3])
@pytest.mark.parametrize("apx", [-1, 0, 1])
@pytest.mark.parametrize("hermitian", [False, True])
def test_trace_matrix_log(dim: int, mk: int, apx: int, hermitian: bool):
    """Like CVXQUAD ``test_trace_logm``: SDP bound and accuracy vs ``tr(C logm(A))``."""
    if mk == 1 and apx == 0:
        pytest.skip("CVXQUAD skips (m,k)=(1,1) with Pade apx=0.")

    seed = dim * 100_003 + mk * 17 + (apx + 1) * 3 + int(hermitian)
    mat_a = _rand_psd_normalized(dim, seed, hermitian=hermitian)
    mat_c = _rand_psd(dim, seed + 1, hermitian=hermitian)

    assert is_positive_semidefinite(np.asarray(mat_a, dtype=np.complex128))
    assert is_positive_semidefinite(np.asarray(mat_c, dtype=np.complex128))

    tr_ref = float(np.real(np.trace(mat_c @ logm(mat_a))))

    np.testing.assert_allclose(
        trace_matrix_log(mat_a, mat_c),
        tr_ref,
        rtol=1e-8,
        atol=1e-8,
    )

    cvx_val, status = _trace_matrix_log_sdp_at_fixed_a(mat_a, mat_c, m=mk, k=mk, apx=apx)
    assert status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, status
    assert cvx_val is not None

    if abs(tr_ref) < 1e-12:
        assert abs(cvx_val - tr_ref) <= 1e-4
        return

    relerr = (cvx_val - tr_ref) / abs(tr_ref)
    # SCS can slightly overshoot the analytic bound on larger dims.
    assert apx * relerr >= -1e-3, relerr
    if mk >= 3:
        assert abs(relerr) <= 1e-2, relerr


def test_trace_matrix_log_mat_c_none_uses_identity() -> None:
    """``mat_c is None`` sets ``C = I`` (numeric path)."""
    n = 3
    rng = np.random.default_rng(7)
    g = rng.standard_normal((n, n))
    mat_x = g @ g.T + float(n) * np.eye(n)
    mat_x = (mat_x + mat_x.T) / 2
    expected = float(np.real(np.trace(logm(mat_x))))
    got = trace_matrix_log(mat_x, None)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_trace_matrix_log_constant_cvx_expression() -> None:
    """Constant CVXPY expression folds to the NumPy path."""
    n = 2
    rng = np.random.default_rng(11)
    g = rng.standard_normal((n, n))
    mat_a = g @ g.T + np.eye(n)
    mat_a = (mat_a + mat_a.T) / 2
    mat_c = np.eye(n)
    expr = cvxpy.Constant(mat_a)
    got = trace_matrix_log(expr, mat_c)
    want = trace_matrix_log(mat_a, mat_c)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


def test_trace_matrix_log_rejects_nonconstant_affine() -> None:
    """Non-constant affine expressions are rejected; use the hypo cone for composition."""
    n = 3
    rng = np.random.default_rng(13)
    g = rng.standard_normal((n, n))
    a0 = g @ g.T + 0.4 * np.eye(n)
    a0 = (a0 + a0.T) / 2
    mat_a = 0.35 * a0
    assert is_positive_semidefinite(mat_a)
    w_var = cvxpy.Variable((n, n), symmetric=True)
    w_var.value = np.zeros((n, n))
    mat_x = cvxpy.Constant(mat_a) + w_var - w_var
    assert mat_x.is_affine() and not mat_x.is_constant()
    with pytest.raises(ValueError, match=_NOT_SUPPORTED):
        trace_matrix_log(mat_x, np.eye(n))


def test_trace_matrix_log_rejects_hermitian_affine() -> None:
    """Hermitian non-constant affine expressions are rejected."""
    n = 2
    rng = np.random.default_rng(17)
    g = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h0 = g @ g.conj().T + 0.5 * np.eye(n)
    h0 = (h0 + h0.conj().T) / 2
    mat_a = 0.3 * h0
    assert is_positive_semidefinite(np.asarray(mat_a, dtype=np.complex128))
    w_var = cvxpy.Variable((n, n), hermitian=True)
    w_var.value = np.zeros((n, n), dtype=np.complex128)
    mat_x = cvxpy.Constant(mat_a) + w_var - w_var
    assert mat_x.is_affine() and not mat_x.is_constant()
    with pytest.raises(ValueError, match=_NOT_SUPPORTED):
        trace_matrix_log(mat_x, np.eye(n))


class TestTraceMatrixLogValueErrors:
    """``ValueError`` paths in ``trace_matrix_log``."""

    def test_mat_x_wrong_type(self) -> None:
        """Reject ``mat_x`` that is neither a NumPy array nor a CVXPY expression."""
        with pytest.raises(
            ValueError,
            match=re.escape("mat_x must be a numpy array or a cvxpy expression"),
        ):
            trace_matrix_log([[1.0, 0.0], [0.0, 1.0]], None)

    def test_mat_x_not_2d(self) -> None:
        """Reject ``mat_x`` that is not two-dimensional."""
        mat_x = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match=re.escape("mat_x must be 2D.")):
            trace_matrix_log(mat_x, None)

    def test_mat_x_not_square(self) -> None:
        """Reject non-square ``mat_x``."""
        mat_x = np.zeros((2, 3))
        with pytest.raises(ValueError, match=re.escape("mat_x must be square.")):
            trace_matrix_log(mat_x, None)

    def test_mat_c_wrong_type(self) -> None:
        """Reject ``mat_c`` that is not a NumPy array."""
        mat_x = np.eye(2)
        mat_c = cvxpy.Constant(np.eye(2))
        with pytest.raises(ValueError, match=re.escape("mat_c must be a numpy array")):
            trace_matrix_log(mat_x, mat_c)

    def test_mat_c_not_2d(self) -> None:
        """Reject ``mat_c`` that is not two-dimensional."""
        mat_x = np.eye(2)
        mat_c = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match=re.escape("mat_c must be 2D.")):
            trace_matrix_log(mat_x, mat_c)

    def test_mat_c_not_square(self) -> None:
        """Reject non-square ``mat_c``."""
        mat_x = np.eye(2)
        mat_c = np.zeros((2, 3))
        with pytest.raises(ValueError, match=re.escape("mat_c must be square.")):
            trace_matrix_log(mat_x, mat_c)

    def test_mat_c_not_psd(self) -> None:
        """Reject ``mat_c`` that is not positive semidefinite."""
        mat_x = np.eye(2)
        mat_c = np.diag([1.0, -0.5])
        with pytest.raises(
            ValueError,
            match=re.escape("mat_c must be a positive semidefinite matrix"),
        ):
            trace_matrix_log(mat_x, mat_c)

    def test_shape_mismatch(self) -> None:
        """Reject ``mat_x`` and ``mat_c`` with different shapes."""
        mat_x = np.eye(2)
        mat_c = np.eye(3)
        with pytest.raises(
            ValueError,
            match=re.escape("mat_x and mat_c must have the same shape"),
        ):
            trace_matrix_log(mat_x, mat_c)

    def test_mat_x_numpy_not_psd(self) -> None:
        """Reject non-PSD NumPy ``mat_x``."""
        mat_x = np.array([[1.0, 2.0], [2.0, 1.0]])
        mat_c = np.eye(2)
        with pytest.raises(
            ValueError,
            match=re.escape("mat_x must be a positive semidefinite matrix"),
        ):
            trace_matrix_log(mat_x, mat_c)

    def test_parameter_no_value(self) -> None:
        """Unset ``Parameter.value`` is rejected."""
        n = 2
        p = cvxpy.Parameter((n, n), symmetric=True)
        assert p.value is None
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass mat_x as a numpy.ndarray."
            ),
        ):
            trace_matrix_log(p, np.eye(n))

    def test_nonconstant_quadratic(self) -> None:
        """Reject non-constant quadratic CVXPY inputs."""
        n = 2
        x_var = cvxpy.Variable((n, n), symmetric=True)
        x_var.value = np.eye(n)
        with pytest.raises(ValueError, match=_NOT_SUPPORTED):
            trace_matrix_log(cvxpy.square(x_var), np.eye(n))

    def test_nonconstant_variable(self) -> None:
        """Reject non-constant CVXPY variables."""
        n = 2
        t = cvxpy.Variable()
        expr = t * np.eye(n)
        with pytest.raises(ValueError, match=_NOT_SUPPORTED):
            trace_matrix_log(expr, np.eye(n))


def test_trace_matrix_log_numpy_still_works():
    """Numeric numpy path is unaffected by the guard."""
    mat_x = np.diag([0.7, 0.3])
    result = trace_matrix_log(mat_x)
    assert np.isfinite(result)


def test_trace_matrix_log_constant_cvxpy_still_works():
    """A CVXPY Constant (no free variables) must not be rejected."""
    mat_x = np.diag([0.7, 0.3])
    result = trace_matrix_log(cvxpy.Constant(mat_x))
    assert np.isfinite(result)


def test_trace_matrix_log_free_variable_raises():
    """A free CVXPY Variable with ``.value`` set is rejected (use the hypo cone)."""
    x_var = cvxpy.Variable((2, 2), symmetric=True)
    x_var.value = np.diag([0.7, 0.3])
    with pytest.raises(ValueError, match=_NOT_SUPPORTED):
        trace_matrix_log(x_var, np.eye(2))
