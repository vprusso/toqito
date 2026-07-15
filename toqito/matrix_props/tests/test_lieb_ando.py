"""Tests for the lieb_ando function."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import re
from fractions import Fraction

import cvxpy
import numpy as np
import pytest
from scipy.linalg import fractional_matrix_power

from toqito.cones._utils import _AFFINE_VARIABLE_USE_CONE
from toqito.matrix_props import lieb_ando

_NOT_SUPPORTED = re.escape(_AFFINE_VARIABLE_USE_CONE)

DIMS = (2, 3)
TVEC = (0.5, 0.25, 0.75, 0.125, 1.5, 1.25)
PD_SHIFT = 1e-1

I_2 = np.eye(2)
I_3 = np.eye(3)
I_4 = np.eye(4)


def _case_seed(dim: int, t: float, *, hermitian: bool) -> int:
    r = Fraction(float(t)).limit_denominator()
    return int(dim * 1_000_003 + r.numerator * 10_009 + r.denominator * 100 + int(hermitian))


def _random_pd_matrix(dim: int, seed: int, *, hermitian: bool) -> np.ndarray:
    """Well-conditioned PSD matrix by construction."""
    rng = np.random.default_rng(seed)
    if hermitian:
        x_mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    else:
        x_mat = rng.standard_normal((dim, dim))
    mat = x_mat @ x_mat.conj().T + PD_SHIFT * np.eye(dim, dtype=x_mat.dtype)
    return (mat + mat.conj().T) / 2


def _random_pd_normalized(dim: int, seed: int, *, hermitian: bool) -> np.ndarray:
    """Trace-normalized PSD matrix."""
    m = _random_pd_matrix(dim, seed, hermitian=hermitian)
    tr = np.real(np.trace(m))
    return m / tr


def _numeric_lieb_ando_reference(mat_a: np.ndarray, mat_b: np.ndarray, mat_k: np.ndarray, t: float) -> float:
    r"""Match lieb_ando numeric branch: real(trace(K^H @ A^{1-t} @ K @ B^t))."""
    a_sym = (mat_a + mat_a.conj().T) / 2
    b_sym = (mat_b + mat_b.conj().T) / 2
    pow_a = fractional_matrix_power(a_sym, 1.0 - float(t))
    pow_b = fractional_matrix_power(b_sym, float(t))
    return float(np.real(np.trace(mat_k.conj().T @ pow_a @ mat_k @ pow_b)))


@pytest.mark.parametrize(
    ("mat_a", "mat_b", "mat_k", "t", "expected_msg"),
    [
        (np.zeros((3,)), I_3, I_3, 0.5, "mat_a must be 2D."),
        (I_3, np.zeros((3,)), I_3, 0.5, "mat_b must be 2D."),
        (np.zeros((4, 2)), I_4, np.eye(4, 4), 0.5, "mat_a must be square."),
        (I_2, np.ones((2, 3)), np.ones((2, 3)), 0.5, "mat_b must be square."),
        (I_2, I_2, np.zeros((3,)), 0.5, "mat_k must be 2D."),
        (
            I_2,
            I_2,
            np.eye(2, 3),
            0.5,
            "mat_k must have the same number of rows as mat_a and the same number of columns as mat_b.",
        ),
        (
            I_2,
            I_2,
            np.eye(3, 2),
            0.5,
            "mat_k must have the same number of rows as mat_a and the same number of columns as mat_b.",
        ),
        (
            np.array([[1.0, 0.0], [0.0, -1.0]]),
            I_2,
            I_2,
            0.5,
            "mat_a and mat_b must be positive semidefinite.",
        ),
        (
            I_2,
            np.array([[1.0, 0.0], [0.0, -1.0]]),
            I_2,
            0.5,
            "mat_a and mat_b must be positive semidefinite.",
        ),
    ],
)
def test_lieb_ando_raises(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    mat_k: np.ndarray,
    t: float,
    expected_msg: str,
):
    """lieb_ando rejects invalid shapes and indefinite numeric PSD inputs."""
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        lieb_ando(mat_a, mat_b, mat_k, t)


def test_lieb_ando_raises_type_error_mat_k_not_numpy():
    """mat_k must be a numpy.ndarray (not a CVXPY expression)."""
    with pytest.raises(
        TypeError,
        match=re.escape("mat_k must be a numpy.ndarray."),
    ):
        lieb_ando(I_2, I_2, cvxpy.Constant(I_2), 0.5)


def test_lieb_ando_raises_type_error_mat_a_not_numpy_or_cvx_expression():
    """mat_a must be a numpy.ndarray or cvxpy.Expression."""

    class _Square2DNonNumeric:
        __slots__ = ()
        ndim = 2
        shape = (2, 2)

    with pytest.raises(
        TypeError,
        match=re.escape("mat_a must be a numpy.ndarray or a cvxpy expression."),
    ):
        lieb_ando(_Square2DNonNumeric(), I_2, I_2, 0.5)


def test_lieb_ando_raises_type_error_mat_b_not_numpy_or_cvx_expression():
    """mat_b must be a numpy.ndarray or cvxpy.Expression."""

    class _Square2DNonNumeric:
        __slots__ = ()
        ndim = 2
        shape = (2, 2)

    with pytest.raises(
        TypeError,
        match=re.escape("mat_b must be a numpy.ndarray or a cvxpy expression."),
    ):
        lieb_ando(I_2, _Square2DNonNumeric(), I_2, 0.5)


@pytest.mark.parametrize(
    ("mat_a_expr", "mat_b_expr", "mat_k", "t", "expected_msg"),
    [
        (
            cvxpy.Constant(np.ones((2, 3))),
            cvxpy.Constant(I_3),
            np.ones((2, 3)),
            0.5,
            "mat_a must be square.",
        ),
        (
            cvxpy.Constant(I_2),
            cvxpy.Constant(np.ones((2, 3))),
            np.ones((2, 3)),
            0.5,
            "mat_b must be square.",
        ),
        (
            cvxpy.Constant(I_2),
            cvxpy.Constant(I_2),
            np.ones((2, 3)),
            0.5,
            "mat_k must have the same number of rows as mat_a and the same number of columns as mat_b.",
        ),
        (
            cvxpy.Constant(np.diag([-0.1, 0.5])),
            cvxpy.Constant(I_2),
            I_2,
            0.5,
            "mat_a and mat_b must be positive semidefinite.",
        ),
        (
            cvxpy.Constant(I_2),
            cvxpy.Constant(np.diag([-0.1, 0.5])),
            I_2,
            0.5,
            "mat_a and mat_b must be positive semidefinite.",
        ),
    ],
)
def test_lieb_ando_raises_cv_expression_shapes_and_psd(
    mat_a_expr: cvxpy.Expression,
    mat_b_expr: cvxpy.Expression,
    mat_k: np.ndarray,
    t: float,
    expected_msg: str,
):
    """Shape and PSD checks on constant CVXPY paths."""
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        lieb_ando(mat_a_expr, mat_b_expr, mat_k, t)


def test_lieb_ando_rejects_non_affine():
    """Non-constant (including non-affine) CVXPY inputs are rejected."""
    var_x = cvxpy.Variable((2, 2), symmetric=True)
    non_affine = var_x @ var_x
    with pytest.raises(ValueError, match=_NOT_SUPPORTED):
        lieb_ando(non_affine, cvxpy.Constant(I_2), I_2, 0.5)
    with pytest.raises(ValueError, match=_NOT_SUPPORTED):
        lieb_ando(cvxpy.Constant(I_2), non_affine, I_2, 0.5)
    with pytest.raises(ValueError, match=_NOT_SUPPORTED):
        lieb_ando(non_affine, non_affine, I_2, 0.5)


def test_lieb_ando_raises_psd_mixed_numeric_a_expression_b():
    """Numeric mat_a PSD but mat_b constant value not PSD."""
    psd_a = np.array([[4.0, 2.0], [2.0, 1.0]])
    bad_b = cvxpy.Constant(np.diag([-0.05, 0.2]))
    with pytest.raises(
        ValueError,
        match=re.escape("mat_a and mat_b must be positive semidefinite."),
    ):
        lieb_ando(psd_a, bad_b, I_2, 0.5)


def test_lieb_ando_raises_psd_mixed_expression_a_numeric_b():
    """Test mat_a constant value not PSD but mat_b numeric PSD."""
    bad_a = cvxpy.Constant(np.diag([-0.05, 0.2]))
    psd_b = np.array([[4.0, 2.0], [2.0, 1.0]])
    with pytest.raises(
        ValueError,
        match=re.escape("mat_a and mat_b must be positive semidefinite."),
    ):
        lieb_ando(bad_a, psd_b, I_2, 0.5)


def test_lieb_ando_numeric_a_constant_b_matches_reference():
    """Numeric mat_a and Constant mat_b unwrap to the numeric path."""
    mat_a = np.array([[4.0, 2.0], [2.0, 1.0]], dtype=float)
    mat_b_np = np.diag([0.6, 0.9])
    mat_b = cvxpy.Constant(mat_b_np)
    mat_k = np.eye(2)
    for t in (0.25, 0.75):
        ref = _numeric_lieb_ando_reference(mat_a, mat_b_np, mat_k, t)
        val = float(np.real(lieb_ando(mat_a, mat_b, mat_k, t)))
        np.testing.assert_allclose(val, ref, rtol=1e-5, atol=1e-5)


def test_lieb_ando_constant_a_numeric_b_matches_reference():
    """Constant mat_a and numeric mat_b unwrap to the numeric path."""
    mat_a_np = np.diag([1.5, 0.8])
    mat_a = cvxpy.Constant(mat_a_np)
    mat_b = np.diag([0.6, 1.1])
    mat_k = np.eye(2)
    for t in (0.25, 0.75):
        ref = _numeric_lieb_ando_reference(mat_a_np, mat_b, mat_k, t)
        val = float(np.real(lieb_ando(mat_a, mat_b, mat_k, t)))
        np.testing.assert_allclose(val, ref, rtol=1e-5, atol=1e-5)


def test_lieb_ando_numeric_a_constant_b_epi_region():
    """Mixed numeric/Constant evaluation for t outside [0, 1]."""
    rng = np.random.default_rng(7)
    n = 3
    mat_a = rng.standard_normal((n, n))
    mat_a = mat_a @ mat_a.T + np.eye(n)
    mat_b_np = rng.standard_normal((n, n))
    mat_b_np = mat_b_np @ mat_b_np.T + np.eye(n)
    mat_b = cvxpy.Constant(mat_b_np)
    mat_k = np.eye(n)
    t = 1.35
    ref = _numeric_lieb_ando_reference(mat_a, mat_b_np, mat_k, t)
    val = float(np.real(lieb_ando(mat_a, mat_b, mat_k, t)))
    np.testing.assert_allclose(val, ref, rtol=1e-5, atol=1e-5)


def test_lieb_ando_constant_a_numeric_b_epi_region():
    """Mixed Constant/numeric evaluation for t outside [0, 1]."""
    rng = np.random.default_rng(8)
    n = 3
    mat_a_np = rng.standard_normal((n, n))
    mat_a_np = mat_a_np @ mat_a_np.T + np.eye(n)
    mat_a = cvxpy.Constant(mat_a_np)
    mat_b = rng.standard_normal((n, n))
    mat_b = mat_b @ mat_b.T + np.eye(n)
    mat_k = np.eye(n)
    t = -0.4
    ref = _numeric_lieb_ando_reference(mat_a_np, mat_b, mat_k, t)
    val = float(np.real(lieb_ando(mat_a, mat_b, mat_k, t)))
    np.testing.assert_allclose(val, ref, rtol=1e-5, atol=1e-5)


def test_lieb_ando_numeric_matches_scipy_reference_small():
    """Closed form vs reference on a tiny PSD pair."""
    mat_a = np.array([[4.0, 2.0], [2.0, 1.0]], dtype=float)
    mat_b = np.diag([0.5, 1.5])
    mat_k = np.array([[1.0, 0.0], [0.0, 1.0]])
    t = 0.3
    ref = _numeric_lieb_ando_reference(mat_a, mat_b, mat_k, t)
    val = lieb_ando(mat_a, mat_b, mat_k, t)
    assert float(np.real(val)) == pytest.approx(ref)


def test_lieb_ando_numeric_matches_scipy_identity_k():
    """K = I matches reference for random trace-normalized PSD."""
    rng = np.random.default_rng(0)
    n = 3
    mat_a = rng.standard_normal((n, n))
    mat_a = mat_a @ mat_a.T + 0.3 * np.eye(n)
    mat_a = mat_a / np.trace(mat_a)
    mat_b = rng.standard_normal((n, n))
    mat_b = mat_b @ mat_b.T + 0.3 * np.eye(n)
    mat_b = mat_b / np.trace(mat_b)
    in_n = np.eye(n)
    for t in (0.2, 0.8, 1.25, -0.25):
        ref = _numeric_lieb_ando_reference(mat_a, mat_b, in_n, t)
        val = lieb_ando(mat_a, mat_b, in_n, t)
        assert float(np.real(val)) == pytest.approx(ref, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("t", TVEC)
@pytest.mark.parametrize("hermitian", [False, True])
def test_lieb_ando_cvxquad_numeric_grid_trace_normalized(
    dim: int,
    t: float,
    hermitian: bool,
):
    """Grid from cvxquad_tests.m test_lieb_ando: nvec, tvec, real/complex."""
    seed = _case_seed(dim, t, hermitian=hermitian)
    mat_a = _random_pd_normalized(dim, seed, hermitian=hermitian)
    mat_b = _random_pd_normalized(dim, seed + 1, hermitian=hermitian)
    in_n = np.eye(dim, dtype=mat_a.dtype)
    ref = _numeric_lieb_ando_reference(mat_a, mat_b, in_n, t)
    val = float(np.real(lieb_ando(mat_a, mat_b, in_n, t)))
    np.testing.assert_allclose(val, ref, rtol=1e-10, atol=1e-10)


def test_lieb_ando_numeric_complex_rectangular_k():
    """Rectangular K with Hermitian PSD A, B (reference matches numeric branch)."""
    rng = np.random.default_rng(2)
    n, m = 2, 3
    xa = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    mat_a = xa @ xa.conj().T + PD_SHIFT * np.eye(n, dtype=np.complex128)
    mat_a = (mat_a + mat_a.conj().T) / 2
    xb = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    mat_b = xb @ xb.conj().T + PD_SHIFT * np.eye(m, dtype=np.complex128)
    mat_b = (mat_b + mat_b.conj().T) / 2
    mat_k = (rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))) * 0.3
    t = 0.4
    ref = _numeric_lieb_ando_reference(mat_a, mat_b, mat_k, t)
    val = float(np.real(lieb_ando(mat_a, mat_b, mat_k, t)))
    np.testing.assert_allclose(val, ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("t", TVEC)
@pytest.mark.parametrize("hermitian", [False, True])
def test_lieb_ando_constant_expressions_match_numeric(
    dim: int,
    t: float,
    hermitian: bool,
):
    """cvxpy.Constant(A), Constant(B) vs numeric reference."""
    seed = _case_seed(dim, t, hermitian=hermitian)
    mat_a = _random_pd_normalized(dim, seed, hermitian=hermitian)
    mat_b = _random_pd_normalized(dim, seed + 1, hermitian=hermitian)
    in_n = np.eye(dim, dtype=mat_a.dtype)
    ref = _numeric_lieb_ando_reference(mat_a, mat_b, in_n, t)
    val = float(np.real(lieb_ando(cvxpy.Constant(mat_a), cvxpy.Constant(mat_b), in_n, t)))
    atol = 5e-4 if 0 <= t <= 1 else 5e-3
    np.testing.assert_allclose(val, ref, rtol=1e-4, atol=atol)


@pytest.mark.parametrize("t", (0.25, 0.6, -0.35, 1.3))
def test_lieb_ando_constant_matches_numeric_complex_nontrivial_k(t: float):
    """Constant unwrap agrees with numeric when K is genuinely complex."""
    rng = np.random.default_rng(101)
    n = 2
    xa = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    mat_a = xa @ xa.conj().T + 0.6 * np.eye(n, dtype=np.complex128)
    mat_a = (mat_a + mat_a.conj().T) / 2
    xb = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    mat_b = xb @ xb.conj().T + 0.6 * np.eye(n, dtype=np.complex128)
    mat_b = (mat_b + mat_b.conj().T) / 2
    mat_k = np.array(
        [[0.35 + 0.22j, 0.12 - 0.31j], [-0.18 + 0.09j, 0.42 + 0.28j]],
        dtype=np.complex128,
    )

    ref = float(np.real(lieb_ando(mat_a, mat_b, mat_k, t)))
    val = float(np.real(lieb_ando(cvxpy.Constant(mat_a), cvxpy.Constant(mat_b), mat_k, t)))
    np.testing.assert_allclose(val, ref, rtol=1e-10, atol=1e-10)


def test_lieb_ando_numpy_still_works():
    """Numeric numpy path is unaffected by the guard."""
    mat_a = np.diag([0.7, 0.3])
    mat_b = np.diag([0.6, 0.4])
    mat_k = np.eye(2)
    result = lieb_ando(mat_a, mat_b, mat_k, 0.5)
    assert result > 0


def test_lieb_ando_constant_cvxpy_still_works():
    """CVXPY Constants (no free variables) must not be rejected."""
    mat_a = np.diag([0.7, 0.3])
    mat_b = np.diag([0.6, 0.4])
    mat_k = np.eye(2)
    result = lieb_ando(cvxpy.Constant(mat_a), cvxpy.Constant(mat_b), mat_k, 0.5)
    assert result > 0


def test_lieb_ando_constant_mat_a_no_value():
    """Reject constant ``mat_a`` with no numeric ``.value``."""
    p_a = cvxpy.Parameter((2, 2), symmetric=True)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Constant CVXPY expression has no numeric value; set parameter `.value` "
            "or pass mat_a as a numpy.ndarray."
        ),
    ):
        lieb_ando(p_a, I_2, I_2, 0.5)


def test_lieb_ando_constant_mat_b_no_value():
    """Reject constant ``mat_b`` with no numeric ``.value``."""
    p_b = cvxpy.Parameter((2, 2), symmetric=True)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Constant CVXPY expression has no numeric value; set parameter `.value` "
            "or pass mat_b as a numpy.ndarray."
        ),
    ):
        lieb_ando(I_2, p_b, I_2, 0.5)


def test_lieb_ando_free_variable_mat_a_raises():
    """A free CVXPY Variable in mat_a must raise the shared nonconstant guard message."""
    mat_b = np.diag([0.6, 0.4])
    mat_k = np.eye(2)
    x_var = cvxpy.Variable((2, 2), symmetric=True)
    x_var.value = np.diag([0.7, 0.3])
    with pytest.raises(
        ValueError,
        match=_NOT_SUPPORTED,
    ):
        lieb_ando(x_var, cvxpy.Constant(mat_b), mat_k, 0.5)


def test_lieb_ando_free_variable_mat_b_raises():
    """A free CVXPY Variable in mat_b must raise the shared nonconstant guard message."""
    mat_a = np.diag([0.7, 0.3])
    mat_k = np.eye(2)
    y_var = cvxpy.Variable((2, 2), symmetric=True)
    y_var.value = np.diag([0.6, 0.4])
    with pytest.raises(
        ValueError,
        match=_NOT_SUPPORTED,
    ):
        lieb_ando(cvxpy.Constant(mat_a), y_var, mat_k, 0.5)
