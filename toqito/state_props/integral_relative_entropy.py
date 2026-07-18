r"""Integral-representation SDPs for quantum relative entropy [@kossmann2024optimisingrelativeentropy]."""

import warnings
from typing import Any

import cvxpy
import numpy as np

from toqito.cones._integral_relative_entropy_helpers import (
    _generalized_eigenvalues,
    _make_delta,
    _make_gamma,
    _make_grid,
    _sandwich_parameters,
)
from toqito.cones._utils import _reject_nonconstant_cvxpy
from toqito.cones.integral_relative_entropy_lower_cone import (
    integral_relative_entropy_lower_cone,
)
from toqito.cones.integral_relative_entropy_upper_cone import (
    integral_relative_entropy_upper_cone,
)
from toqito.matrix_props import is_positive_semidefinite

# Re-export helpers used by channel_metrics and tests.
__all__ = [
    "evaluate_relative_entropy_integral",
    "_generalized_eigenvalues",
    "_make_delta",
    "_make_gamma",
    "_make_grid",
    "_sandwich_parameters",
]


def evaluate_relative_entropy_integral(
    mat_x: np.ndarray | cvxpy.Expression,
    mat_y: np.ndarray | cvxpy.Expression,
    *,
    epsilon_dec: float = 1e-2,
    mean: bool = True,
    solver: str = "SCS",
    **solve_kwargs: Any,
) -> float | tuple[float, float]:
    r"""Estimate \(D(X\|Y)\) via integral SDP lower/upper bounds [@kossmann2024optimisingrelativeentropy].

    For fixed PSD matrices \(X\) and \(Y\), discretizes

    \[
        D(X\|Y) = \int_\mu^\lambda \frac{ds}{s}\,\mathrm{tr}^+[Y s - X]
        + \log\lambda + 1 - \lambda
    \]

    and returns the midpoint of the resulting lower and upper semidefinite bounds.
    All auxiliary SDPs use \(n \times n\) matrices (dimension of ``mat_x``).
    Affine or variable CVXPY inputs are not supported; use
    ``integral_relative_entropy_lower_cone`` /
    ``integral_relative_entropy_upper_cone`` for composition (with explicit
    ``mu`` / ``lam`` when values are not fixed).

    Args:
        mat_x: The first positive semidefinite matrix \(X\).
        mat_y: The second positive semidefinite matrix \(Y\).
        epsilon_dec: Grid refinement parameter \(\varepsilon\).
        mean: If ``True``, return the average of the bounds; otherwise ``(lower, upper)``.
        solver: CVXPY solver name.
        solve_kwargs: Extra arguments for ``cvxpy.Problem.solve``.

    Returns:
        Either the midpoint estimate or the pair ``(lower, upper)``.

    Raises:
        ValueError: If inputs contain unsupported free CVXPY variables,
            are not PSD, shapes mismatch, or \(\mu,\lambda\) are
            degenerate.
        RuntimeError: If a bound SDP fails to solve.

    """
    if isinstance(mat_x, cvxpy.Expression) and mat_x.value is not None:
        mat_x_eval = np.asarray(mat_x.value)
    else:
        mat_x_eval = np.asarray(mat_x)
    if isinstance(mat_y, cvxpy.Expression) and mat_y.value is not None:
        mat_y_eval = np.asarray(mat_y.value)
    else:
        mat_y_eval = np.asarray(mat_y)
    if mat_x.shape != mat_y.shape:
        raise ValueError("mat_x and mat_y must have the same shape.")
    if not is_positive_semidefinite(mat_x_eval):
        raise ValueError("mat_x must be a positive semidefinite matrix.")
    if not is_positive_semidefinite(mat_y_eval):
        raise ValueError("mat_y must be a positive semidefinite matrix.")
    _reject_nonconstant_cvxpy(mat_x, mat_y)
    if np.allclose(mat_x_eval, mat_y_eval):
        if mean:
            return 0.0
        return 0.0, 0.0

    is_cplx = np.iscomplexobj(mat_x_eval) or np.iscomplexobj(mat_y_eval)
    default_kwargs = {"eps": 1e-8, "verbose": False}
    default_kwargs.update(solve_kwargs)

    mu, lam = _sandwich_parameters(mat_x_eval, mat_y_eval)

    x_c = cvxpy.Constant(mat_x_eval)
    y_c = cvxpy.Constant(mat_y_eval)

    t_lower = cvxpy.Variable()
    lower_cons = integral_relative_entropy_lower_cone(
        x_c,
        y_c,
        t_lower,
        epsilon_dec=epsilon_dec,
        mu=mu,
        lam=lam,
        hermitian=is_cplx,
    )
    lower_prob = cvxpy.Problem(cvxpy.Minimize(t_lower), lower_cons)
    lower_prob.solve(solver=solver, **default_kwargs)
    if lower_prob.status not in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Lower-bound SDP failed: {lower_prob.status}")
    if lower_prob.status == cvxpy.OPTIMAL_INACCURATE:
        warnings.warn("Lower-bound SDP returned OPTIMAL_INACCURATE; result may be off.")
    lower = float(np.real(lower_prob.value))

    t_upper = cvxpy.Variable()
    upper_cons = integral_relative_entropy_upper_cone(
        x_c,
        y_c,
        t_upper,
        epsilon_dec=epsilon_dec,
        mu=mu,
        lam=lam,
        hermitian=is_cplx,
    )
    upper_prob = cvxpy.Problem(cvxpy.Minimize(t_upper), upper_cons)
    upper_prob.solve(solver=solver, **default_kwargs)
    if upper_prob.status not in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Upper-bound SDP failed: {upper_prob.status}")
    if upper_prob.status == cvxpy.OPTIMAL_INACCURATE:
        warnings.warn("Upper-bound SDP returned OPTIMAL_INACCURATE; result may be off.")
    upper = float(np.real(upper_prob.value))

    if mean:
        return (lower + upper) / 2
    return lower, upper
