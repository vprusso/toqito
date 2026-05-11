"""Tests for the lieb_ando function."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import re
from fractions import Fraction

import cvxpy
import numpy as np
import pytest
from scipy.linalg import fractional_matrix_power

from toqito.matrix_ops import lieb_ando

DIMS = (2, 3)
TVEC = (0.5, 0.25, 0.75, 0.125, 1.5, 1.25)
PD_SHIFT = 1e-1

I_2 = np.eye(2)
I_3 = np.eye(3)
I_4 = np.eye(4)


def _case_seed(dim: int, t: float, *, hermitian: bool) -> int:
    r = Fraction(float(t)).limit_denominator()
    return int(
        dim * 1_000_003 + r.numerator * 10_009 + r.denominator * 100 + int(hermitian)
    )


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


def _numeric_lieb_ando_reference(
    mat_a: np.ndarray, mat_b: np.ndarray, mat_k: np.ndarray, t: float
) -> float:
    r"""Match lieb_ando numeric branch: real(trace(K.T @ A^{1-t} @ K @ B^t))."""
    a_sym = (mat_a + mat_a.conj().T) / 2
    b_sym = (mat_b + mat_b.conj().T) / 2
    pow_a = fractional_matrix_power(a_sym, 1.0 - float(t))
    pow_b = fractional_matrix_power(b_sym, float(t))
    return float(np.real(np.trace(mat_k.T @ pow_a @ mat_k @ pow_b)))


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
        (
            cvxpy.Constant(I_2),
            cvxpy.Constant(I_2),
            I_2,
            2.2,
            "t must be between -1 and 2.",
        ),
        (
            cvxpy.Constant(I_2),
            cvxpy.Constant(I_2),
            I_2,
            -1.2,
            "t must be between -1 and 2.",
        ),
    ],
)
def test_lieb_ando_raises_cv_expression_shapes_and_psd_and_t(
    mat_a_expr: cvxpy.Expression,
    mat_b_expr: cvxpy.Expression,
    mat_k: np.ndarray,
    t: float,
    expected_msg: str,
):
    """Shape, PSD, and t range checks on CVXPY paths."""
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        lieb_ando(mat_a_expr, mat_b_expr, mat_k, t)


def test_lieb_ando_raises_non_affine_both_expressions():
    """Both arguments must be affine when they are CVXPY expressions."""
    var_x = cvxpy.Variable((2, 2), symmetric=True)
    non_affine = var_x @ var_x
    with pytest.raises(
        ValueError,
        match=re.escape("mat_a and mat_b must be affine expressions."),
    ):
        lieb_ando(non_affine, cvxpy.Constant(I_2), I_2, 0.5)
    with pytest.raises(
        ValueError,
        match=re.escape("mat_a and mat_b must be affine expressions."),
    ):
        lieb_ando(cvxpy.Constant(I_2), non_affine, I_2, 0.5)
    with pytest.raises(
        ValueError,
        match=re.escape("mat_a and mat_b must be affine expressions."),
    ):
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
    """Numeric mat_a and affine mat_b (trace_power reduction branch).

    Keep t in (0, 1) so mat_kak stays numerically PSD for trace_power.
    """
    mat_a = np.array([[4.0, 2.0], [2.0, 1.0]], dtype=float)
    mat_b_np = np.diag([0.6, 0.9])
    mat_b = cvxpy.Constant(mat_b_np)
    mat_k = np.eye(2)
    for t in (0.25, 0.75):
        ref = _numeric_lieb_ando_reference(mat_a, mat_b_np, mat_k, t)
        val = float(np.real(lieb_ando(mat_a, mat_b, mat_k, t)))
        np.testing.assert_allclose(val, ref, rtol=1e-5, atol=1e-5)


def test_lieb_ando_constant_a_numeric_b_matches_reference():
    """Affine mat_a and numeric mat_b (trace_power reduction branch).

    Diagonal A, B avoids stiff coupled SDPs inside trace_power.
    """
    mat_a_np = np.diag([1.5, 0.8])
    mat_a = cvxpy.Constant(mat_a_np)
    mat_b = np.diag([0.6, 1.1])
    mat_k = np.eye(2)
    for t in (0.25, 0.75):
        ref = _numeric_lieb_ando_reference(mat_a_np, mat_b, mat_k, t)
        val = float(np.real(lieb_ando(mat_a, mat_b, mat_k, t)))
        np.testing.assert_allclose(val, ref, rtol=1e-3, atol=5e-3)


def test_lieb_ando_numeric_a_constant_b_epi_region_trace_power():
    """Test t outside [0, 1] on mixed branch: well-conditioned data for trace_power epi."""
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
    np.testing.assert_allclose(val, ref, rtol=2e-2, atol=2e-2)


def test_lieb_ando_constant_a_numeric_b_epi_region_trace_power():
    """Epi-region t on affine-A / numeric-B branch."""
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
    np.testing.assert_allclose(val, ref, rtol=2e-2, atol=2e-2)


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
    val = float(
        np.real(lieb_ando(cvxpy.Constant(mat_a), cvxpy.Constant(mat_b), in_n, t))
    )
    atol = 5e-4 if 0 <= t <= 1 else 5e-3
    np.testing.assert_allclose(val, ref, rtol=1e-4, atol=atol)
