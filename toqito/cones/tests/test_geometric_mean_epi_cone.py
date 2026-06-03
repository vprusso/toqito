"""Tests for geometric_mean_epi_cone."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import re
from fractions import Fraction

import cvxpy
import numpy as np
import pytest

from toqito.cones.geometric_mean import geometric_mean
from toqito.cones.geometric_mean_epi_cone import geometric_mean_epi_cone

DIMS = (3, 5)
# t in [-1, 0] (epi branch t <= 0) and [1, 2] (branch t >= 1); disjoint from hypo (0, 1).
EPI_WEIGHTS_NEG = (-1.0, -0.75, -0.5, -1 / 3, -0.25, 0.0)
EPI_WEIGHTS_POS = (1.0, 1.25, 4 / 3, 1.5, 5 / 3, 1.75, 2.0)
EPI_WEIGHTS = EPI_WEIGHTS_NEG + EPI_WEIGHTS_POS
PD_SHIFT = 1e-1


def _case_seed(dim: int, t: float, *, hermitian: bool) -> int:
    r = Fraction(float(t)).limit_denominator()
    return int(dim * 1_000_003 + r.numerator * 10_009 + r.denominator * 100 + int(hermitian))


def _random_pd_matrix(dim: int, seed: int, *, hermitian: bool) -> np.ndarray:
    """Generate a well-conditioned positive-definite matrix by construction."""
    rng = np.random.default_rng(seed)
    if hermitian:
        x_mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    else:
        x_mat = rng.standard_normal((dim, dim))
    mat = x_mat @ x_mat.conj().T + PD_SHIFT * np.eye(dim, dtype=x_mat.dtype)
    return (mat + mat.conj().T) / 2


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("w", EPI_WEIGHTS)
@pytest.mark.parametrize("hermitian", [False, True])
def test_geometric_mean_epi_cone_trace_minimum(dim: int, w: float, hermitian: bool):
    """Epigraph SDP recovers geometric mean via trace minimization (dual of hypo trace max)."""
    seed = _case_seed(dim, w, hermitian=hermitian)
    a_np = _random_pd_matrix(dim, seed, hermitian=hermitian)
    b_np = _random_pd_matrix(dim, seed + 1, hermitian=hermitian)

    reference = geometric_mean(a_np, b_np, float(w))

    a_const = cvxpy.Constant(a_np)
    b_const = cvxpy.Constant(b_np)
    if hermitian:
        t_var = cvxpy.Variable((dim, dim), hermitian=True)
    else:
        t_var = cvxpy.Variable((dim, dim), symmetric=True)

    constraints = geometric_mean_epi_cone(
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
    problem.solve(solver=cvxpy.SCS, eps=1e-8, max_iters=400_000, verbose=False)

    assert problem.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, problem.status
    assert t_var.value is not None

    rtol = 2.5e-2
    atol = 1e-2
    np.testing.assert_allclose(t_var.value, reference, rtol=rtol, atol=atol)


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
            "The weight must be in the range [-1, 0] or [1, 2].",
        ),
        (
            cvxpy.Constant(I_2),
            cvxpy.Constant(I_2),
            cvxpy.Constant(I_2),
            -1.1,
            "The weight must be in the range [-1, 0] or [1, 2].",
        ),
        (
            cvxpy.Constant(I_2),
            cvxpy.Constant(I_2),
            cvxpy.Constant(I_2),
            2.1,
            "The weight must be in the range [-1, 0] or [1, 2].",
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
def test_geometric_mean_epi_cone_invalid_input(
    a_expr: cvxpy.Expression,
    b_expr: cvxpy.Expression,
    t_expr: cvxpy.Expression,
    t_weight: float,
    expected_msg: str,
):
    """``geometric_mean_epi_cone`` raises ``ValueError`` for invalid arguments."""
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        geometric_mean_epi_cone(a_expr, b_expr, t_expr, t_weight, hermitian=False)
