"""Tests for geometric_mean_hypo_cone."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import re
from fractions import Fraction

import cvxpy
import numpy as np
import pytest

from toqito.matrix_ops.geometric_mean import geometric_mean
from toqito.matrix_ops.geometric_mean_hypo_cone import geometric_mean_hypo_cone

DIMS = (3, 5)
WEIGHTS = (0.5, 0.25, 0.125, 0.0625, 0.75, 0.875, 0.9375, 2 / 3, 6 / 7)
PD_SHIFT = 1e-1


def _case_seed(dim: int, t: float, *, hermitian: bool) -> int:
    r = Fraction(float(t)).limit_denominator()
    return int(
        dim * 1_000_003 + r.numerator * 10_009 + r.denominator * 100 + int(hermitian)
    )


def _random_pd_matrix(dim: int, seed: int, *, hermitian: bool) -> np.ndarray:
    """Generate a well-conditioned positive-definite matrix by construction."""
    rng = np.random.default_rng(seed)
    if hermitian:
        x_mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    else:
        x_mat = rng.standard_normal((dim, dim))
    return x_mat @ x_mat.conj().T + PD_SHIFT * np.eye(dim, dtype=x_mat.dtype)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("w", WEIGHTS)
@pytest.mark.parametrize("hermitian", [False, True])
def test_geometric_mean_hypo_cone_trace_maximum(dim: int, w: float, hermitian: bool):
    """Hypograph SDP recovers geometric mean via trace maximization (like MATLAB test)."""
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

    constraints = geometric_mean_hypo_cone(
        a_const,
        b_const,
        t_var,
        w,
        fullhyp=False,
        hermitian=hermitian,
    )
    obj = cvxpy.trace(t_var)
    if hermitian:
        obj = cvxpy.real(obj)
    problem = cvxpy.Problem(cvxpy.Maximize(obj), constraints)
    problem.solve(solver=cvxpy.SCS, verbose=False)

    assert problem.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, problem.status
    assert t_var.value is not None

    rtol = 1e-6
    atol = 5e-4
    np.testing.assert_allclose(t_var.value, reference, rtol=rtol, atol=atol)


def test_geometric_mean_hypo_cone_fullhyp_feasibility_matlab_example():
    """Same feasibility check as CVXQUAD cvxquad_tests (fullhyp=1, t=1/2, n=2)."""
    a_mat = np.array([[6.25, 0.0], [0.0, 16.0]])
    b_mat = np.array([[2.0, 1.0], [1.0, 2.0]])
    eye2 = np.eye(2)
    t = 0.5
    constraints = geometric_mean_hypo_cone(
        cvxpy.Constant(a_mat),
        cvxpy.Constant(eye2),
        cvxpy.Constant(b_mat),
        t,
        hermitian=False,
    )
    problem = cvxpy.Problem(cvxpy.Minimize(0), constraints)
    problem.solve(solver=cvxpy.SCS, verbose=False)
    assert problem.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, problem.status
    assert problem.value is not None
    np.testing.assert_allclose(float(problem.value), 0.0, atol=1e-6)


@pytest.mark.parametrize("endpoint_t", [0.0, 1.0])
def test_geometric_mean_hypo_cone_weight_endpoints_restricted(
    endpoint_t: float,
):
    """Recursion base cases ``t == 0`` and ``t == 1`` (feasible constant triple)."""
    dim = 2
    rng = np.random.default_rng(42)
    a_np = rng.standard_normal((dim, dim))
    a_np = a_np @ a_np.T + 0.5 * np.eye(dim)
    b_np = rng.standard_normal((dim, dim))
    b_np = b_np @ b_np.T + 0.5 * np.eye(dim)
    t_np = a_np if endpoint_t == 0.0 else b_np
    constraints = geometric_mean_hypo_cone(
        cvxpy.Constant(a_np),
        cvxpy.Constant(b_np),
        cvxpy.Constant(t_np),
        endpoint_t,
        fullhyp=False,
        hermitian=False,
    )
    problem = cvxpy.Problem(cvxpy.Minimize(0), constraints)
    problem.solve(solver=cvxpy.SCS, verbose=False)
    assert problem.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, problem.status


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
            -0.05,
            "The weight must be in the range [0, 1].",
        ),
        (
            cvxpy.Constant(I_2),
            cvxpy.Constant(I_2),
            cvxpy.Constant(I_2),
            1.01,
            "The weight must be in the range [0, 1].",
        ),
        (
            cvxpy.Constant(A_2),
            cvxpy.Constant(B_3),
            cvxpy.Constant(I_2),
            0.5,
            "The matrices must be the same size.",
        ),
        (
            cvxpy.Constant(np.eye(3)),
            cvxpy.Constant(np.eye(3)),
            cvxpy.Constant(I_2),
            0.5,
            "The matrices must be the same size.",
        ),
        (
            cvxpy.Constant(A_rect),
            cvxpy.Constant(A_rect),
            cvxpy.Constant(A_rect),
            0.5,
            "The matrices must be square.",
        ),
    ],
)
def test_geometric_mean_hypo_cone_invalid_input(
    a_expr: cvxpy.Expression,
    b_expr: cvxpy.Expression,
    t_expr: cvxpy.Expression,
    t_weight: float,
    expected_msg: str,
):
    """``geometric_mean_hypo_cone`` raises ``ValueError`` for invalid arguments."""
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        geometric_mean_hypo_cone(
            a_expr, b_expr, t_expr, t_weight, fullhyp=False, hermitian=False
        )
