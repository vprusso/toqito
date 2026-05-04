"""Tests for matrix_geo_mean_epi_cone."""

import re
from fractions import Fraction

import cvxpy
import numpy as np
import pytest

from toqito.matrix_ops.matrix_geo_mean import matrix_geo_mean
from toqito.matrix_ops.matrix_geo_mean_epi_cone import matrix_geo_mean_epi_cone
from toqito.rand import random_psd_operator

DIMS = (3, 5)
# t in [-1, 0] (epi branch t <= 0) and [1, 2] (branch t >= 1); disjoint from hypo (0, 1).
EPI_WEIGHTS_NEG = (-1.0, -0.75, -0.5, -1 / 3, -0.25, 0.0)
EPI_WEIGHTS_POS = (1.0, 1.25, 4 / 3, 1.5, 5 / 3, 1.75, 2.0)
EPI_WEIGHTS = EPI_WEIGHTS_NEG + EPI_WEIGHTS_POS
PD_JITTER = 1e-6


def _case_seed(dim: int, t: float, *, hermitian: bool) -> int:
    r = Fraction(float(t)).limit_denominator()
    return int(
        dim * 1_000_003 + r.numerator * 10_009 + r.denominator * 100 + int(hermitian)
    )


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("w", EPI_WEIGHTS)
@pytest.mark.parametrize("hermitian", [False, True])
def test_matrix_geo_mean_epi_cone_trace_minimum(
    dim: int, w: float, hermitian: bool
) -> None:
    """Epigraph SDP recovers geometric mean via trace minimization (dual of hypo trace max)."""
    seed = _case_seed(dim, w, hermitian=hermitian)
    a_mat = random_psd_operator(
        dim,
        is_real=not hermitian,
        seed=seed,
        distribution="wishart",
        num_degrees=dim + 4,
    )
    b_mat = random_psd_operator(
        dim,
        is_real=not hermitian,
        seed=seed + 1,
        distribution="wishart",
        num_degrees=dim + 4,
    )
    jitter = PD_JITTER * np.eye(dim, dtype=a_mat.dtype)
    a_np = a_mat + jitter
    b_np = b_mat + jitter

    reference = matrix_geo_mean(a_np, b_np, float(w))

    a_const = cvxpy.Constant(a_np)
    b_const = cvxpy.Constant(b_np)
    if hermitian:
        t_var = cvxpy.Variable((dim, dim), hermitian=True)
    else:
        t_var = cvxpy.Variable((dim, dim), symmetric=True)

    constraints = matrix_geo_mean_epi_cone(
        a_const,
        b_const,
        t_var,
        w,
        hermitian=hermitian,
    )
    obj = cvxpy.trace(t_var)
    if hermitian:
        obj = cvxpy.real(obj)
    problem = cvxpy.Problem(cvxpy.Minimize(obj), constraints)
    problem.solve(solver=cvxpy.SCS, verbose=False)

    assert problem.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, problem.status
    assert t_var.value is not None

    tol = 1e-2
    np.testing.assert_allclose(t_var.value, reference, rtol=1e-5, atol=tol)


I_2 = np.eye(2)
A_2 = np.array([[2.0, 0.1], [0.1, 1.0]])
B_3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
A_rect = np.ones((2, 3))


@pytest.mark.parametrize(
    "a_expr, b_expr, t_expr, t_weight, expected_msg",
    [
        (
            cvxpy.Constant(I_2),
            cvxpy.Constant(I_2),
            cvxpy.Constant(I_2),
            0.5,
            "t has to be in [-1,0] or [1,2]",
        ),
        (
            cvxpy.Constant(I_2),
            cvxpy.Constant(I_2),
            cvxpy.Constant(I_2),
            -1.1,
            "t has to be in [-1,0] or [1,2]",
        ),
        (
            cvxpy.Constant(I_2),
            cvxpy.Constant(I_2),
            cvxpy.Constant(I_2),
            2.1,
            "t has to be in [-1,0] or [1,2]",
        ),
        (
            cvxpy.Constant(A_2),
            cvxpy.Constant(B_3),
            cvxpy.Constant(I_2),
            -0.5,
            "The matrices must be the same size.",
        ),
        (
            cvxpy.Constant(np.eye(3)),
            cvxpy.Constant(np.eye(3)),
            cvxpy.Constant(I_2),
            -0.5,
            "The matrices must be the same size.",
        ),
        (
            cvxpy.Constant(A_rect),
            cvxpy.Constant(A_rect),
            cvxpy.Constant(A_rect),
            -0.5,
            "The matrices must be square.",
        ),
    ],
)
def test_matrix_geo_mean_epi_cone_invalid_input(
    a_expr: cvxpy.Expression,
    b_expr: cvxpy.Expression,
    t_expr: cvxpy.Expression,
    t_weight: float,
    expected_msg: str,
) -> None:
    """``matrix_geo_mean_epi_cone`` raises ``ValueError`` for invalid arguments."""
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        matrix_geo_mean_epi_cone(a_expr, b_expr, t_expr, t_weight, hermitian=False)
