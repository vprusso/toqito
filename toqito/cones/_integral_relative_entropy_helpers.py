"""Numeric helpers for integral relative-entropy sandwich grids."""

import cvxpy
import numpy as np
from scipy.linalg import LinAlgError, eig, eigh


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
    """
    grid = [mu]
    curr = mu + np.sqrt(epsilon * mu * 8)
    while curr < lam:
        grid.append(curr)
        curr = curr + np.sqrt(epsilon * 8 * curr)
    return np.array(grid + [lam])


def _make_delta(t: np.ndarray) -> np.ndarray:
    r"""Make the delta coefficients for the integral representation."""
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
    r"""Make the gamma coefficients for the integral representation."""
    gamma = np.zeros(len(t))
    gamma[0] = -1 * ((1 + t[0] / (t[1] - t[0])) * np.log(t[1] / t[0]) - 1)
    gamma[-1] = -1 * (1 - t[-2] * np.log(t[-1] / t[-2]) / (t[-1] - t[-2]))
    for i in range(1, len(t) - 1):
        gamma[i] = -1 * (
            (1 + t[i] / (t[i + 1] - t[i])) * np.log(t[i + 1] / t[i])
            - np.log(t[i] / t[i - 1]) * t[i - 1] / (t[i] - t[i - 1])
        )
    return gamma


def _integral_correction(lam: float) -> float:
    return float(np.log(lam) + 1 - lam)


def _require_valid_sandwich(mu: float, lam: float) -> None:
    if mu <= 0 or lam <= mu:
        raise ValueError(
            "The integral representation requires 0 < mu < lambda. "
            "This typically means the matrices are too close for the bound "
            "(support of X may not be contained in support of Y)."
        )


def _numeric_pair_for_sandwich(
    mat_x: cvxpy.Expression,
    mat_y: cvxpy.Expression,
) -> tuple[np.ndarray, np.ndarray]:
    if mat_x.value is None or mat_y.value is None:
        raise ValueError(
            "Sandwich parameters require numeric `.value` on mat_x and mat_y, "
            "or pass mu and lam explicitly."
        )
    return np.asarray(mat_x.value), np.asarray(mat_y.value)
