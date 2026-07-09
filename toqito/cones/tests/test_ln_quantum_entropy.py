"""Tests for ln_quantum_entropy."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import re

import cvxpy
import numpy as np
import pytest
from scipy.linalg import logm

from toqito.cones.ln_quantum_entropy import ln_quantum_entropy
from toqito.cones.operator_relative_entropy_epi_cone import (
    operator_relative_entropy_epi_cone,
)
from toqito.matrix_props import is_positive_semidefinite


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


def _ln_quantum_entropy_reference(mat_x: np.ndarray) -> float:
    """Exact quantum entropy in nats."""
    return float(np.real(-np.trace(mat_x @ logm(mat_x))))


def _ln_quantum_entropy_sdp_at_fixed_x(
    mat_x: np.ndarray,
    *,
    m: int,
    k: int,
    apx: int,
) -> tuple[float, str]:
    """SDP for ``ln_quantum_entropy`` at fixed ``X = mat_x`` (CVXQUAD ``X == A`` case)."""
    n = mat_x.shape[0]
    is_cplx = np.any(np.imag(mat_x) != 0)
    if is_cplx:
        tau = cvxpy.Variable((n, n), hermitian=True)
    else:
        tau = cvxpy.Variable((n, n), symmetric=True)
    eye_n = cvxpy.Constant(np.eye(n))
    x_c = cvxpy.Constant(mat_x)
    cons = operator_relative_entropy_epi_cone(
        x_c,
        eye_n,
        tau,
        m=m,
        k=k,
        e=np.eye(n),
        apx=-apx,
        hermitian=is_cplx,
    )
    obj = -cvxpy.trace(tau)
    if is_cplx:
        obj = cvxpy.real(obj)
    prob = cvxpy.Problem(cvxpy.Maximize(obj), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    return val, prob.status


@pytest.mark.parametrize("dim", [3, 5, 8])
@pytest.mark.parametrize("mk", [1, 3])
@pytest.mark.parametrize("apx", [-1, 0, 1])
@pytest.mark.parametrize("hermitian", [False, True])
def test_ln_quantum_entropy(dim: int, mk: int, apx: int, hermitian: bool):
    """Like CVXQUAD ``test_quantum_entr``: SDP bound and accuracy vs exact entropy."""
    if mk == 1 and apx == 0:
        pytest.skip("CVXQUAD skips (m,k)=(1,1) with Pade apx=0.")

    seed = dim * 100_003 + mk * 17 + (apx + 1) * 3 + int(hermitian)
    mat_a = _rand_psd_normalized(dim, seed, hermitian=hermitian)

    assert is_positive_semidefinite(np.asarray(mat_a, dtype=np.complex128))

    h_ref = _ln_quantum_entropy_reference(mat_a)
    np.testing.assert_allclose(
        ln_quantum_entropy(mat_a, m=mk, k=mk, apx=apx),
        h_ref,
        rtol=1e-8,
        atol=1e-8,
    )

    cvx_val, status = _ln_quantum_entropy_sdp_at_fixed_x(mat_a, m=mk, k=mk, apx=apx)
    assert status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, status
    assert cvx_val is not None

    if abs(h_ref) < 1e-12:
        assert abs(cvx_val - h_ref) <= 1e-4
        return

    relerr = (cvx_val - h_ref) / abs(h_ref)
    assert apx * relerr >= -1e-4, relerr
    if mk >= 3:
        assert abs(relerr) <= 1e-2, relerr


def test_ln_quantum_entropy_maximally_mixed():
    """Sanity check on ``I/d`` (entropy in nats): ``-tr(rho log rho) = log(d)``."""
    dim = 4
    mat_x = np.eye(dim) / dim
    expected = float(np.log(dim))
    got = ln_quantum_entropy(mat_x)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_ln_quantum_entropy_constant_cvx_expression():
    """Constant CVXPY expression folds to the NumPy path."""
    n = 2
    rng = np.random.default_rng(11)
    g = rng.standard_normal((n, n))
    mat_a = g @ g.T + np.eye(n)
    mat_a = (mat_a + mat_a.T) / 2
    mat_a = mat_a / np.trace(mat_a)
    expr = cvxpy.Constant(mat_a)
    got = ln_quantum_entropy(expr, m=3, k=3, apx=0)
    want = ln_quantum_entropy(mat_a, m=3, k=3, apx=0)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


def test_ln_quantum_entropy_affine_real_sdp():
    """Non-constant affine ``mat_x`` hits the cone SDP; ``Constant(A)+W-W`` is algebraically ``A``."""
    n = 3
    rng = np.random.default_rng(13)
    g = rng.standard_normal((n, n))
    a0 = g @ g.T + 0.4 * np.eye(n)
    a0 = (a0 + a0.T) / 2
    mat_a = a0 / np.trace(a0)
    assert is_positive_semidefinite(mat_a)
    w_var = cvxpy.Variable((n, n), symmetric=True)
    w_var.value = np.zeros((n, n))
    mat_x = cvxpy.Constant(mat_a) + w_var - w_var
    assert mat_x.is_affine() and not mat_x.is_constant()
    val = ln_quantum_entropy(mat_x, m=3, k=3, apx=1)
    ref = _ln_quantum_entropy_reference(mat_a)
    np.testing.assert_allclose(float(val), ref, rtol=5e-4, atol=1e-6)


def test_ln_quantum_entropy_affine_hermitian_sdp():
    """Hermitian ``Constant(A)+W-W`` exercises the complex Hermitian-variable SDP path."""
    n = 2
    rng = np.random.default_rng(17)
    g = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h0 = g @ g.conj().T + 0.5 * np.eye(n)
    h0 = (h0 + h0.conj().T) / 2
    mat_a = h0 / np.trace(h0)
    assert is_positive_semidefinite(np.asarray(mat_a, dtype=np.complex128))
    w_var = cvxpy.Variable((n, n), hermitian=True)
    w_var.value = np.zeros((n, n), dtype=np.complex128)
    mat_x = cvxpy.Constant(mat_a) + w_var - w_var
    assert mat_x.is_affine() and not mat_x.is_constant()
    val = ln_quantum_entropy(mat_x, m=3, k=3, apx=-1)
    ref = _ln_quantum_entropy_reference(mat_a)
    np.testing.assert_allclose(float(val), ref, rtol=5e-4, atol=1e-6)


class TestLnQuantumEntropyValueErrors:
    """``ValueError`` paths in ``ln_quantum_entropy``."""

    def test_mat_x_wrong_type(self):
        """Reject ``mat_x`` that is not a numpy array or CVXPY expression."""
        with pytest.raises(
            ValueError,
            match=re.escape("mat_x must be a numpy array or a cvxpy expression"),
        ):
            ln_quantum_entropy([[1.0, 0.0], [0.0, 1.0]])

    def test_mat_x_not_2d(self):
        """Reject non-2D ``mat_x``."""
        with pytest.raises(ValueError, match=re.escape("mat_x must be 2D.")):
            ln_quantum_entropy(np.array([1.0, 2.0]))

    def test_mat_x_not_square(self):
        """Reject non-square ``mat_x``."""
        with pytest.raises(ValueError, match=re.escape("mat_x must be square.")):
            ln_quantum_entropy(np.zeros((2, 3)))

    def test_m_invalid(self):
        """Reject quadrature count ``m`` below 1."""
        with pytest.raises(ValueError, match=re.escape("m must be at least 1")):
            ln_quantum_entropy(np.eye(2), m=0)

    def test_k_invalid(self):
        """Reject square-root count ``k`` below 1."""
        with pytest.raises(ValueError, match=re.escape("k must be at least 1")):
            ln_quantum_entropy(np.eye(2), k=0)

    def test_apx_invalid(self):
        """Reject approximation flag outside ``{-1, 0, 1}``."""
        with pytest.raises(ValueError, match=re.escape("apx must be -1, 0, or 1")):
            ln_quantum_entropy(np.eye(2), apx=2)

    def test_mat_x_numpy_not_psd(self):
        """Reject non-PSD numeric ``mat_x``."""
        mat_x = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(
            ValueError,
            match=re.escape("mat_x must be a positive semidefinite matrix"),
        ):
            ln_quantum_entropy(mat_x)

    def test_parameter_no_value(self):
        """Reject CVXPY parameters with no ``.value``."""
        n = 2
        p = cvxpy.Parameter((n, n), symmetric=True)
        assert p.value is None
        msg_constant = (
            "Constant CVXPY expression has no numeric value; set parameter `.value` or pass mat_x as a numpy.ndarray."
        )
        msg_affine = "Affine mat_x has no numeric initial value; set `.value` for PSD checks."
        pattern = "|".join((re.escape(msg_constant), re.escape(msg_affine)))
        with pytest.raises(ValueError, match=pattern):
            ln_quantum_entropy(p)

    def test_expression_not_affine(self):
        """Reject non-affine CVXPY ``mat_x``."""
        n = 2
        x_var = cvxpy.Variable((n, n), symmetric=True)
        x_var.value = np.eye(n)
        expr = x_var @ x_var
        assert not expr.is_affine()
        with pytest.raises(
            ValueError,
            match=re.escape("mat_x must be an affine CVXPY expression."),
        ):
            ln_quantum_entropy(expr)

    def test_affine_expression_no_value(self):
        """Reject affine ``mat_x`` with no initial ``.value`` for PSD checks."""
        n = 2
        t = cvxpy.Variable()
        expr = t * np.eye(n)
        assert expr.is_affine()
        assert expr.value is None
        with pytest.raises(
            ValueError,
            match=re.escape("Affine or variable CVXPY inputs are not yet supported; pass numeric matrices."),
        ):
            ln_quantum_entropy(expr)

    def test_affine_expression_not_psd_at_value(self):
        """Reject affine ``mat_x`` that is not PSD at ``.value``."""
        n = 2
        t = cvxpy.Variable()
        expr = t * np.eye(n)
        t.value = -1.0
        assert expr.value is not None
        with pytest.raises(
            ValueError,
            match="Affine or variable CVXPY inputs are not yet supported; pass numeric matrices.",
        ):
            ln_quantum_entropy(expr)


def test_ln_quantum_entropy_numpy_still_works():
    """Numeric numpy path is unaffected by the guard."""
    mat_x = np.diag([0.7, 0.3])
    result = ln_quantum_entropy(mat_x)
    assert np.isfinite(result)


def test_ln_quantum_entropy_constant_cvxpy_still_works():
    """A CVXPY Constant (no free variables) must not be rejected."""
    mat_x = np.diag([0.7, 0.3])
    result = ln_quantum_entropy(cvxpy.Constant(mat_x))
    assert np.isfinite(result)


def test_ln_quantum_entropy_free_variable_raises():
    """A free CVXPY Variable with a valid PSD .value now proceeds through the SDP path."""
    x_var = cvxpy.Variable((2, 2), symmetric=True)
    x_var.value = np.diag([0.7, 0.3])
    val = ln_quantum_entropy(x_var)
    ref = float(np.real(-np.trace(np.diag([0.7, 0.3]) @ logm(np.diag([0.7, 0.3])))))
    np.testing.assert_allclose(float(val), ref, rtol=5e-3, atol=1e-5)
