r"""Integral-representation SDPs for quantum relative entropy [@kossmann2024optimisingrelativeentropy]."""

import warnings

import cvxpy
import numpy as np
from scipy.linalg import LinAlgError, eig, eigh

from toqito.matrix_props import is_positive_semidefinite


def _generalized_eigenvalues(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return real generalized eigenvalues for the pencil ``(a, b)``."""
    try:
        return np.real(eigh(a, b, check_finite=False)[0])
    except LinAlgError:
        return np.real(eig(a, b, left=False, right=False)[0])


def _sandwich_parameters(rho: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    r"""Return sandwich bounds \(\mu\) and \(\lambda\) for PSD matrices \(X\) and \(Y\)."""
    try:
        w_xy = _generalized_eigenvalues(rho, sigma)
        w_yx = _generalized_eigenvalues(sigma, rho)
    except LinAlgError as exc:
        raise ValueError(
            "Failed to compute sandwich parameters from generalized eigenvalues."
        ) from exc
    finite_xy = w_xy[np.isfinite(w_xy)]
    finite_yx = w_yx[np.isfinite(w_yx)]
    if finite_xy.size == 0 or finite_yx.size == 0:
        raise ValueError(
            "Failed to compute sandwich parameters from generalized eigenvalues."
        )
    lam = float(np.max(finite_xy))
    mu = float(np.min(finite_yx))
    return mu, lam


def _make_grid(mu: float, lam: float, epsilon: float) -> np.ndarray:
    r"""Make a grid of points for the integral representation of the relative entropy.

    The first point of the grid is set to \(\mu\). For the k-th point \(t_k\), where
    \(k \gt 1\), \(t_k = t_{k-1} + \sqrt{8 \epsilon t_{k-1}}\).

    This formula yields \(O(\sqrt{\lambda/\epsilon})\) points in the grid.

    Args:
        mu: The starting point of the grid.
        lam: The ending point of the grid.
        epsilon: The grid refinement parameter.

    Returns:
        The grid of points.

    """
    grid = [mu]
    curr = mu + np.sqrt(epsilon * mu * 8)
    while curr < lam:
        grid.append(curr)
        curr = curr + np.sqrt(epsilon * 8 * curr)
    return np.array(grid + [lam])


def _make_delta(t: np.ndarray) -> np.ndarray:
    r"""Make the delta coefficients for the integral representation of the relative entropy.

    Suppose the integral grid has \(r\) points from \(t_1\) to \(t_r\). Then the
    coefficient \(\delta_k\) is defined by

    \[
    \delta_k =
    \begin{cases}
    \left[\left(1 + \frac{t_1}{t_2 - t_1}\right)\log\left(\frac{t_2}{t_1}\right) - 1\right] t_1
    & k = 1, \\
    \left[1 - \frac{t_{r-1}}{t_r - t_{r-1}}\log\left(\frac{t_r}{t_{r-1}}\right)\right] t_r
    & k = r, \\
    \left[\left(1 + \frac{t_k}{t_{k+1} - t_k}\right)\log\left(\frac{t_{k+1}}{t_k}\right)
    - \frac{t_{k-1}}{t_k - t_{k-1}}\log\left(\frac{t_k}{t_{k-1}}\right)\right] t_k
    & \text{otherwise}.
    \end{cases}
    \]

    where the indexing in the formula is one-based.

    Args:
        t: The grid of points.

    Returns:
        The delta coefficients.

    """
    delta = np.zeros(len(t))
    delta[0] = t[0] * ((1 + t[0] / (t[1] - t[0])) * np.log(t[1] / t[0]) - 1)
    delta[-1] = t[-1] * (1 - (np.log(t[-1] / t[-2]) * t[-2] / (t[-1] - t[-2])))
    for i in range(1, len(t) - 1):
        delta[i] = t[i] * (
            (1 + t[i] / (t[i + 1] - t[i])) * np.log(t[i + 1] / t[i])
            - t[i - 1] * np.log(t[i] / t[i - 1]) / (t[i] - t[i - 1])
        )
    return delta


def _make_gamma(t: np.ndarray) -> np.ndarray:
    r"""Make the gamma coefficients for the integral representation of the relative entropy.

    Suppose the integral grid has \(r\) points from \(t_1\) to \(t_r\). Then the
    coefficient \(\gamma_k\) is defined by

    \[
    \gamma_k =
    \begin{cases}
    -\left[\left(1 + \frac{t_1}{t_2 - t_1}\right)\log\left(\frac{t_2}{t_1}\right) - 1\right]
    & k = 1, \\
    -\left[1 - \frac{t_{r-1}}{t_r - t_{r-1}}\log\left(\frac{t_r}{t_{r-1}}\right)\right]
    & k = r, \\
    -\left[\left(1 + \frac{t_k}{t_{k+1} - t_k}\right)\log\left(\frac{t_{k+1}}{t_k}\right)
    - \frac{t_{k-1}}{t_k - t_{k-1}}\log\left(\frac{t_k}{t_{k-1}}\right)\right]
    & \text{otherwise}.
    \end{cases}
    \]

    where the indexing in the formula is one-based.

    Args:
        t: The grid of points.

    Returns:
        The gamma coefficients.

    """
    gamma = np.zeros(len(t))
    gamma[0] = -1 * ((1 + t[0] / (t[1] - t[0])) * np.log(t[1] / t[0]) - 1)
    gamma[-1] = -1 * (1 - t[-2] * np.log(t[-1] / t[-2]) / (t[-1] - t[-2]))
    for i in range(1, len(t) - 1):
        gamma[i] = -1 * (
            (1 + t[i] / (t[i + 1] - t[i])) * np.log(t[i + 1] / t[i])
            - np.log(t[i] / t[i - 1]) * t[i - 1] / (t[i] - t[i - 1])
        )
    return gamma


def _matrix_variable(n: int, *, complex_hermitian: bool) -> cvxpy.Variable:
    if complex_hermitian:
        return cvxpy.Variable((n, n), hermitian=True)
    return cvxpy.Variable((n, n), symmetric=True)


def _integral_correction(lam: float) -> float:
    return float(np.log(lam) + 1 - lam)


def evaluate_relative_entropy_integral(
    mat_x: np.ndarray,
    mat_y: np.ndarray,
    *,
    epsilon_dec: float = 1e-2,
    mean: bool = True,
    solver: str = "SCS",
    **solve_kwargs,
) -> float | tuple[float, float]:
    r"""Estimate \(D(X\|Y)\) via integral SDP lower/upper bounds [@kossmann2024optimisingrelativeentropy].

    For fixed PSD matrices \(X\) and \(Y\), discretizes

    \[
        D(X\|Y) = \int_\mu^\lambda \frac{ds}{s}\,\mathrm{tr}^+[Y s - X]
        + \log\lambda + 1 - \lambda
    \]

    and returns the midpoint of the resulting lower and upper semidefinite bounds.
    All auxiliary SDPs use \(n \times n\) matrices (dimension of ``mat_x``).

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
        ValueError: If inputs are not PSD, shapes mismatch, or \(\mu,\lambda\) are degenerate.
        RuntimeError: If a bound SDP fails to solve.

    """
    mat_x = np.asarray(mat_x)
    mat_y = np.asarray(mat_y)
    if mat_x.shape != mat_y.shape:
        raise ValueError("mat_x and mat_y must have the same shape.")
    if not is_positive_semidefinite(mat_x):
        raise ValueError("mat_x must be a positive semidefinite matrix.")
    if not is_positive_semidefinite(mat_y):
        raise ValueError("mat_y must be a positive semidefinite matrix.")
    if np.allclose(mat_x, mat_y):
        if mean:
            return 0.0
        return 0.0, 0.0

    n = int(mat_x.shape[0])
    is_cplx = np.iscomplexobj(mat_x) or np.iscomplexobj(mat_y)
    default_kwargs = {"eps": 1e-8, "verbose": False}
    default_kwargs.update(solve_kwargs)

    mu, lam = _sandwich_parameters(mat_x, mat_y)
    if mu <= 0 or lam <= mu:
        raise ValueError(
            "The integral representation requires 0 < mu < lambda. "
            "This typically means the matrices are too close for the bound "
            "(support of X may not be contained in support of Y)."
        )

    t = _make_grid(mu, lam, epsilon_dec)
    r = len(t)
    alpha = [np.log(t[k] / t[k + 1]) for k in range(r - 1)]
    beta = [t[k + 1] - t[k] for k in range(r - 1)]
    gamma = _make_gamma(t)
    delta = _make_delta(t)

    mu_vars = [_matrix_variable(n, complex_hermitian=is_cplx) for _ in range(r - 1)]
    lower_cons = [mu_vars[k] >> 0 for k in range(r - 1)] + [
        mu_vars[k] - alpha[k] * mat_x - beta[k] * mat_y >> 0 for k in range(r - 1)
    ]
    lower_prob = cvxpy.Problem(
        cvxpy.Minimize(cvxpy.sum([cvxpy.trace(mu_vars[k]) for k in range(r - 1)])),
        lower_cons,
    )
    lower_prob.solve(solver=solver, **default_kwargs)
    if lower_prob.status not in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Lower-bound SDP failed: {lower_prob.status}")
    if lower_prob.status == cvxpy.OPTIMAL_INACCURATE:
        warnings.warn("Lower-bound SDP returned OPTIMAL_INACCURATE; result may be off.")
    lower = float(np.real(lower_prob.value)) + _integral_correction(lam)

    nu_vars = [_matrix_variable(n, complex_hermitian=is_cplx) for _ in range(r)]
    upper_cons = [nu_vars[k] >> 0 for k in range(r)] + [
        nu_vars[k] - gamma[k] * mat_x - delta[k] * mat_y >> 0 for k in range(r)
    ]
    upper_prob = cvxpy.Problem(
        cvxpy.Minimize(cvxpy.sum([cvxpy.trace(nu_vars[k]) for k in range(r)])),
        upper_cons,
    )
    upper_prob.solve(solver=solver, **default_kwargs)
    if upper_prob.status not in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Upper-bound SDP failed: {upper_prob.status}")
    if upper_prob.status == cvxpy.OPTIMAL_INACCURATE:
        warnings.warn("Upper-bound SDP returned OPTIMAL_INACCURATE; result may be off.")
    upper = float(np.real(upper_prob.value)) + _integral_correction(lam)

    if mean:
        return (lower + upper) / 2
    return lower, upper
