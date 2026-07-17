"""CVXPY constraints for the integral upper approximation of quantum relative entropy."""

import cvxpy

from toqito.cones._integral_relative_entropy_helpers import (
    _integral_correction,
    _make_delta,
    _make_gamma,
    _make_grid,
    _numeric_pair_for_sandwich,
    _require_valid_sandwich,
    _sandwich_parameters,
)
from toqito.cones._utils import _require_square_2d, _symmetric_like_variable


def integral_relative_entropy_upper_cone(
    mat_x: cvxpy.Expression,
    mat_y: cvxpy.Expression,
    t: cvxpy.Expression,
    *,
    epsilon_dec: float = 1e-2,
    mu: float | None = None,
    lam: float | None = None,
    hermitian: bool = False,
) -> list[cvxpy.Constraint]:
    r"""Return CVXPY constraints for the integral upper approximation of \(D(X\|Y)\).

    Discretizes the integral representation of quantum relative entropy
    [@kossmann2024optimisingrelativeentropy] and enforces

    \[
        t \geqslant U_{\varepsilon}(X, Y),
    \]

    where \(U_{\varepsilon}\) is the upper SDP bound (with \(n \times n\) auxiliaries,
    not the \(n^2\) Kronecker lift of ``quantum_relative_entropy_epi_cone``).

    Sandwich endpoints \(\mu, \lambda\) are taken from ``mu`` / ``lam`` when provided;
    otherwise they are computed from ``mat_x.value`` and ``mat_y.value``.

    Args:
        mat_x: A CVXPY expression for an ``n x n`` PSD matrix \(X\).
        mat_y: A CVXPY expression for an ``n x n`` PSD matrix \(Y\).
        t: A CVXPY scalar (or ``1 x 1``) epigraph variable.
        epsilon_dec: Grid refinement parameter \(\varepsilon\).
        mu: Optional sandwich lower endpoint.
        lam: Optional sandwich upper endpoint.
        hermitian: Whether the matrices are Hermitian or symmetric.

    Raises:
        ValueError: If ``mat_x`` or ``mat_y`` is not square 2D.
        ValueError: If shapes differ, sandwich is degenerate, or numeric values
            are missing when ``mu`` / ``lam`` are omitted.

    Returns:
        A list of CVXPY constraints.

    """
    _require_square_2d(mat_x, "mat_x")
    _require_square_2d(mat_y, "mat_y")
    if mat_x.shape != mat_y.shape:
        raise ValueError("mat_x and mat_y must have the same shape")

    if mu is None or lam is None:
        if mu is not None or lam is not None:
            raise ValueError("mu and lam must both be provided or both omitted")
        x_val, y_val = _numeric_pair_for_sandwich(mat_x, mat_y)
        mu, lam = _sandwich_parameters(x_val, y_val)
    _require_valid_sandwich(mu, lam)

    grid = _make_grid(mu, lam, epsilon_dec)
    gamma = _make_gamma(grid)
    delta = _make_delta(grid)
    r = len(grid)

    n = int(mat_x.shape[0])
    nu_vars = [_symmetric_like_variable(n, hermitian=hermitian) for _ in range(r)]
    constraints: list[cvxpy.Constraint] = [nu_vars[k] >> 0 for k in range(r)]
    constraints.extend(
        nu_vars[k] - float(gamma[k]) * mat_x - float(delta[k]) * mat_y >> 0
        for k in range(r)
    )
    bound = cvxpy.sum([cvxpy.trace(nu_vars[k]) for k in range(r)]) + _integral_correction(
        lam
    )
    if hermitian:
        bound = cvxpy.real(bound)
    constraints.append(t >= bound)
    return constraints
