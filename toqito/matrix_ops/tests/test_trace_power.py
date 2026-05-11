"""Tests for trace_power."""

import re
from fractions import Fraction

import cvxpy
import numpy as np
import pytest
from scipy.linalg import fractional_matrix_power

from toqito.matrix_ops import trace_power

DIMS = (3, 5)
WEIGHTS_HYPO = (0.5, 0.25, 0.125, 2 / 3, 6 / 7)
EPI_WEIGHTS_NEG = (-0.75, -0.5, -1 / 3, -0.25)
EPI_WEIGHTS_POS = (1.25, 4 / 3, 1.5, 5 / 3, 1.75)
PD_SHIFT = 1e-1

I_2 = np.eye(2)
I_3 = np.eye(3)


def _case_seed(dim: int, t: float, *, hermitian: bool) -> int:
    r = Fraction(float(t)).limit_denominator()
    return int(
        dim * 1_000_003 + r.numerator * 10_009 + r.denominator * 100 + int(hermitian)
    )


def _random_pd_matrix(dim: int, seed: int, *, hermitian: bool) -> np.ndarray:
    """Generate a well-conditioned positive-semidefinite matrix by construction."""
    rng = np.random.default_rng(seed)
    if hermitian:
        x_mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    else:
        x_mat = rng.standard_normal((dim, dim))
    mat = x_mat @ x_mat.conj().T + PD_SHIFT * np.eye(dim, dtype=x_mat.dtype)
    return (mat + mat.conj().T) / 2


def _numeric_reference(mat_a: np.ndarray, t: float, mat_c: np.ndarray) -> float:
    a_sym = (mat_a + mat_a.conj().T) / 2
    powered = fractional_matrix_power(a_sym, float(t))
    return float(np.real(np.trace(mat_c @ powered)))


@pytest.mark.parametrize(
    ("mat_a", "mat_c", "t", "expected_msg"),
    [
        (np.zeros((3,)), None, 0.5, "mat_a must be 2D."),
        (np.zeros((3, 3)), np.zeros((2,)), 0.5, "mat_c must be 2D."),
        (np.zeros((4, 2)), None, 0.5, "mat_a must be square."),
        (np.eye(2), np.eye(3), 0.5, "The matrices must be the same size."),
        (
            np.array([[1.0, 0.0], [0.0, -1.0]]),
            None,
            0.5,
            "The matrix mat_a must be positive semidefinite.",
        ),
        (
            I_2,
            np.diag([-0.05, 0.1]),
            0.5,
            "The matrix mat_c must be positive semidefinite.",
        ),
        (
            I_3,
            np.diag([-0.01, 0.5, 0.5]),
            0.5,
            "The matrix mat_c must be positive semidefinite.",
        ),
        (
            I_3,
            np.diag([-0.01, 0.5, 0.5]),
            -0.5,
            "The matrix mat_c must be positive semidefinite.",
        ),
        (
            I_2,
            np.ones((2, 3)),
            0.5,
            "mat_c must be square.",
        ),
    ],
)
def test_trace_power_numeric_raises(
    mat_a: np.ndarray,
    mat_c: np.ndarray | None,
    t: float,
    expected_msg: str,
):
    """trace_power rejects invalid numeric matrices."""
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        trace_power(mat_a, t, mat_c)


def test_trace_power_numeric_t_not_restricted_to_minus_one_two():
    """Numeric path allows any t supported by fractional_matrix_power."""
    i3 = np.eye(3)
    for t in (-1.01, 2.01):
        ref = _numeric_reference(i3, t, np.eye(3))
        assert trace_power(i3, t) == pytest.approx(ref)


def test_trace_power_numpy_with_cvx_expression_mat_c_raises_type_error():
    """mat_c must be a numpy array (or None), not a CVXPY expression."""
    msg = "mat_c must be a numpy.ndarray or None."
    with pytest.raises(TypeError, match=re.escape(msg)):
        trace_power(np.eye(2), 0.5, cvxpy.Variable((2, 2)))


@pytest.mark.parametrize(
    ("mat_a_expr", "t", "mat_c", "expected_msg"),
    [
        (
            cvxpy.Constant(I_2),
            2.1,
            I_2,
            "The exponent t must be in the range [-1, 2].",
        ),
        (
            cvxpy.Constant(I_2),
            -1.1,
            I_2,
            "The exponent t must be in the range [-1, 2].",
        ),
        (
            cvxpy.Constant(np.ones((2, 3))),
            0.5,
            I_2,
            "mat_a must be square.",
        ),
        (
            cvxpy.Constant(I_2),
            0.5,
            I_3,
            "The matrices must be the same size.",
        ),
        (
            cvxpy.Constant(np.array([[1.0, 0.0], [0.0, -0.05]])),
            0.5,
            I_2,
            "The matrix mat_a must be positive semidefinite.",
        ),
    ],
)
def test_trace_power_sdp_raises(
    mat_a_expr: cvxpy.Expression,
    t: float,
    mat_c: np.ndarray | None,
    expected_msg: str,
):
    """CVXPY path raises ValueError from invalid SDP arguments."""
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        trace_power(mat_a_expr, t, mat_c)


def test_trace_power_raises_mat_a_neither_ndarray_nor_cvx_expression():
    """Fallback when mat_a is square 2D but not numpy.ndarray or cvxpy.Expression."""

    class _Square2DNonNumeric:
        __slots__ = ()
        ndim = 2
        shape = (2, 2)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The matrix mat_a must be a numpy array or a cvxpy expression."
        ),
    ):
        trace_power(_Square2DNonNumeric(), 0.5)


def test_trace_power_sdp_raises_non_affine():
    """Only affine mat_a is admitted (matching CVXQUAD DCP check)."""
    var_x = cvxpy.Variable((2, 2), symmetric=True)
    non_affine_a = var_x @ var_x
    with pytest.raises(
        ValueError, match=re.escape("The matrix mat_a must be an affine expression.")
    ):
        trace_power(non_affine_a, 0.5)


def test_trace_power_sdp_raises_mat_c_not_numpy():
    """CVXPY mat_a requires mat_c to be a numpy array or None."""
    msg = "mat_c must be a numpy.ndarray or None."
    bad_c = cvxpy.Variable((2, 2), PSD=True)
    with pytest.raises(TypeError, match=re.escape(msg)):
        trace_power(cvxpy.Constant(I_2), 0.5, bad_c)

    with pytest.raises(TypeError, match=re.escape(msg)):
        trace_power(cvxpy.Constant(I_2), 0.5, cvxpy.Constant(I_2))


def test_trace_power_numeric_identity_half():
    """trace(I^{1/2}) = n with default mat_c."""
    n = 4
    result = trace_power(np.eye(n), 0.5)
    assert result == pytest.approx(float(n))


def test_trace_power_numeric_identity_with_c():
    """Weighted trace on a simple PSD pair."""
    ref = _numeric_reference(
        np.array([[4.0, 2.0], [2.0, 1.0]]), 0.8, np.diag([3.0, 0.5])
    )
    val = trace_power(np.array([[4.0, 2.0], [2.0, 1.0]]), 0.8, np.diag([3.0, 0.5]))
    assert val == pytest.approx(ref)


def test_trace_power_numeric_endpoints_t_minus_one_and_two():
    r"""Boundary exponents t \in {-1, 2} match SciPy."""
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((3, 3))
    mat = mat @ mat.T + 0.3 * np.eye(3)
    for t in (-1.0, 2.0):
        ref = _numeric_reference(mat, t, np.eye(3))
        val = trace_power(mat, t)
        assert val == pytest.approx(ref, rel=1e-10, abs=1e-10)


def test_trace_power_numeric_complex_hermitian():
    """Numeric complex Hermitian PSD path."""
    rng = np.random.default_rng(1)
    x_mat = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    mat = x_mat @ x_mat.conj().T + PD_SHIFT * np.eye(2, dtype=np.complex128)
    mat = (mat + mat.conj().T) / 2
    c_mat = np.eye(2, dtype=np.complex128)
    val = trace_power(mat, 0.3, c_mat)
    ref = _numeric_reference(mat, 0.3, c_mat)
    assert val == pytest.approx(ref)


# def _sdp_solve_settings(t: float) -> dict:
#     """Use tighter tolerances outside [0, 1] (mirror test_geometric_mean_epi_cone)."""
#     if t < 0 or t > 1:
#         return dict(solver=cvxpy.SCS, eps=1e-8, max_iters=400_000, verbose=False)
#     return dict(solver=cvxpy.SCS, verbose=False)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("t", WEIGHTS_HYPO)
@pytest.mark.parametrize("hermitian", [False, True])
def test_trace_power_sdp_matches_numeric_hypograph_region(
    dim: int,
    t: float,
    hermitian: bool,
):
    """Maximize tr(mat_c T) on hypo cone."""
    seed = _case_seed(dim, t, hermitian=hermitian)
    mat_a_np = _random_pd_matrix(dim, seed, hermitian=hermitian)
    mat_c_np = _random_pd_matrix(dim, seed + 2, hermitian=hermitian)
    ref = _numeric_reference(mat_a_np, t, mat_c_np)

    mat_a_const = cvxpy.Constant(mat_a_np)
    val = trace_power(mat_a_const, t, mat_c_np)
    atol = 5e-4
    np.testing.assert_allclose(float(val), ref, rtol=1e-5, atol=atol)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize(
    "t",
    EPI_WEIGHTS_NEG + EPI_WEIGHTS_POS,
)
@pytest.mark.parametrize("hermitian", [False, True])
def test_trace_power_sdp_matches_numeric_epigraph_region(
    dim: int,
    t: float,
    hermitian: bool,
):
    """Minimize on epi cone; optimum matches closed form for fixed PSD data."""
    seed = _case_seed(dim, t, hermitian=hermitian)
    mat_a_np = _random_pd_matrix(dim, seed, hermitian=hermitian)
    mat_c_np = _random_pd_matrix(dim, seed + 2, hermitian=hermitian)
    ref = _numeric_reference(mat_a_np, t, mat_c_np)

    mat_a_const = cvxpy.Constant(mat_a_np)
    val = trace_power(mat_a_const, t, mat_c_np)
    np.testing.assert_allclose(float(val), ref, rtol=2.5e-2, atol=1e-2)
