"""Tests for integral relative-entropy lower/upper cones."""

import re

import cvxpy
import numpy as np
import pytest

from toqito.cones._integral_relative_entropy_helpers import _sandwich_parameters
from toqito.cones.integral_relative_entropy_lower_cone import (
    integral_relative_entropy_lower_cone,
)
from toqito.cones.integral_relative_entropy_upper_cone import (
    integral_relative_entropy_upper_cone,
)
from toqito.state_props.integral_relative_entropy import (
    evaluate_relative_entropy_integral,
)
from toqito.state_props.quantum_relative_entropy import quantum_relative_entropy


def _diag_pair() -> tuple[np.ndarray, np.ndarray]:
    return np.diag([0.7, 0.3]), np.diag([0.4, 0.6])


def test_integral_relative_entropy_cones_match_float_bounds():
    """Minimize each cone at Constants and match ``evaluate_...(..., mean=False)``."""
    mat_x, mat_y = _diag_pair()
    ref_lo, ref_hi = evaluate_relative_entropy_integral(mat_x, mat_y, mean=False)

    t_lo = cvxpy.Variable()
    cons_lo = integral_relative_entropy_lower_cone(
        cvxpy.Constant(mat_x), cvxpy.Constant(mat_y), t_lo
    )
    prob_lo = cvxpy.Problem(cvxpy.Minimize(t_lo), cons_lo)
    val_lo = prob_lo.solve(solver=cvxpy.SCS, verbose=False, eps=1e-8)
    assert prob_lo.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob_lo.status
    assert val_lo == pytest.approx(ref_lo, abs=1e-4)

    t_hi = cvxpy.Variable()
    cons_hi = integral_relative_entropy_upper_cone(
        cvxpy.Constant(mat_x), cvxpy.Constant(mat_y), t_hi
    )
    prob_hi = cvxpy.Problem(cvxpy.Minimize(t_hi), cons_hi)
    val_hi = prob_hi.solve(solver=cvxpy.SCS, verbose=False, eps=1e-8)
    assert prob_hi.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob_hi.status
    assert val_hi == pytest.approx(ref_hi, abs=1e-4)
    assert val_hi >= val_lo - 1e-6


def test_integral_relative_entropy_cones_near_quantum_relative_entropy():
    """Integral bounds should sandwich the numeric relative entropy."""
    mat_x, mat_y = _diag_pair()
    ref = quantum_relative_entropy(mat_x, mat_y)
    lo, hi = evaluate_relative_entropy_integral(mat_x, mat_y, mean=False)
    assert lo <= ref + 5e-2
    assert hi >= ref - 5e-2


def test_integral_relative_entropy_cones_explicit_mu_lam():
    """Explicit sandwich endpoints match auto-computed ones."""
    mat_x, mat_y = _diag_pair()
    mu, lam = _sandwich_parameters(mat_x, mat_y)

    t_auto = cvxpy.Variable()
    cons_auto = integral_relative_entropy_lower_cone(
        cvxpy.Constant(mat_x), cvxpy.Constant(mat_y), t_auto
    )
    val_auto = cvxpy.Problem(cvxpy.Minimize(t_auto), cons_auto).solve(
        solver=cvxpy.SCS, verbose=False, eps=1e-8
    )

    t_exp = cvxpy.Variable()
    cons_exp = integral_relative_entropy_lower_cone(
        cvxpy.Constant(mat_x),
        cvxpy.Constant(mat_y),
        t_exp,
        mu=mu,
        lam=lam,
    )
    val_exp = cvxpy.Problem(cvxpy.Minimize(t_exp), cons_exp).solve(
        solver=cvxpy.SCS, verbose=False, eps=1e-8
    )
    assert val_exp == pytest.approx(val_auto, abs=1e-5)


def test_integral_relative_entropy_lower_cone_composition_with_explicit_sandwich():
    """Free ``X`` with fixed ``Y`` and explicit ``mu``/``lam`` remains solvable."""
    mat_x0, mat_y = _diag_pair()
    mu, lam = _sandwich_parameters(mat_x0, mat_y)
    x_var = cvxpy.Variable((2, 2), symmetric=True)
    t = cvxpy.Variable()
    cons = integral_relative_entropy_lower_cone(
        x_var,
        cvxpy.Constant(mat_y),
        t,
        mu=mu,
        lam=lam,
        hermitian=False,
    )
    cons.extend(
        [
            x_var >> 0.05 * np.eye(2),
            x_var << np.eye(2),
            cvxpy.trace(x_var) == 1,
        ]
    )
    prob = cvxpy.Problem(cvxpy.Minimize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False)
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val is not None
    assert np.isfinite(val)


def test_integral_relative_entropy_cones_complex_hermitian():
    """Hermitian Constants use Hermitian auxiliaries."""
    mat_x = np.diag([0.6, 0.4]).astype(complex)
    mat_y = np.diag([0.5, 0.5]).astype(complex)
    t = cvxpy.Variable()
    cons = integral_relative_entropy_upper_cone(
        cvxpy.Constant(mat_x),
        cvxpy.Constant(mat_y),
        t,
        hermitian=True,
    )
    prob = cvxpy.Problem(cvxpy.Minimize(t), cons)
    val = prob.solve(solver=cvxpy.SCS, verbose=False, eps=1e-8)
    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert val == pytest.approx(
        evaluate_relative_entropy_integral(mat_x, mat_y, mean=False)[1],
        abs=1e-3,
    )


def test_integral_relative_entropy_cones_shape_mismatch() -> None:
    """Reject mismatched shapes on both cones."""
    t = cvxpy.Variable()
    for cone in (
        integral_relative_entropy_lower_cone,
        integral_relative_entropy_upper_cone,
    ):
        with pytest.raises(
            ValueError, match=re.escape("mat_x and mat_y must have the same shape")
        ):
            cone(
                cvxpy.Variable((2, 2), symmetric=True),
                cvxpy.Variable((3, 3), symmetric=True),
                t,
                mu=0.1,
                lam=2.0,
            )


def test_integral_relative_entropy_cones_mat_x_not_square() -> None:
    """Reject non-square ``mat_x``."""
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match=re.escape("mat_x must be square.")):
        integral_relative_entropy_upper_cone(
            cvxpy.Variable((2, 3)),
            cvxpy.Variable((2, 2), symmetric=True),
            t,
            mu=0.1,
            lam=2.0,
        )


def test_integral_relative_entropy_cones_partial_mu_lam() -> None:
    """Reject providing only one of ``mu`` / ``lam`` on both cones."""
    mat_x, mat_y = _diag_pair()
    t = cvxpy.Variable()
    for cone in (
        integral_relative_entropy_lower_cone,
        integral_relative_entropy_upper_cone,
    ):
        with pytest.raises(
            ValueError,
            match=re.escape("mu and lam must both be provided or both omitted"),
        ):
            cone(cvxpy.Constant(mat_x), cvxpy.Constant(mat_y), t, mu=0.1)


def test_integral_relative_entropy_cones_missing_value_without_sandwich() -> None:
    """Free Variables without ``mu``/``lam`` cannot compute sandwich endpoints."""
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match="pass mu and lam explicitly"):
        integral_relative_entropy_lower_cone(
            cvxpy.Variable((2, 2), symmetric=True),
            cvxpy.Variable((2, 2), symmetric=True),
            t,
        )


def test_integral_relative_entropy_cones_degenerate_sandwich() -> None:
    """Reject degenerate ``mu`` / ``lam``."""
    mat_x, mat_y = _diag_pair()
    t = cvxpy.Variable()
    with pytest.raises(ValueError, match="0 < mu < lambda"):
        integral_relative_entropy_upper_cone(
            cvxpy.Constant(mat_x),
            cvxpy.Constant(mat_y),
            t,
            mu=1.0,
            lam=1.0,
        )
