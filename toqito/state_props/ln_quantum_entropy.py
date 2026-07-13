r"""Quantum (von Neumann) entropy with natural logarithm: \(-\operatorname{tr}(X \log X)\) for PSD \(X\)."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np
from scipy.linalg import logm

from toqito.cones._utils import _reject_nonconstant_cvxpy, _require_square_2d
from toqito.matrix_props import is_positive_semidefinite


def ln_quantum_entropy(mat_x: np.ndarray | cvxpy.Expression) -> float:
    r"""Compute the quantum entropy \(-\operatorname{tr}(X \log X)\) for PSD \(X\).

    Note that this function uses the natural logarithm (base e) and not the base-2 logarithm.

    This function evaluates the formula numerically. Constant CVXPY expressions
    with a concrete ``.value`` are routed through the numeric path. Affine or
    variable CVXPY inputs are not supported; use
    :func:`~toqito.cones.ln_quantum_entropy_hypo_cone` for composition in a
    parent SDP (with quadrature parameters ``m``, ``k``, ``apx``).

    Args:
        mat_x: The PSD matrix to compute the quantum entropy of, or a constant CVXPY expression.

    Raises:
        ValueError: If mat_x is not a numpy array or a cvxpy expression.
        ValueError: If mat_x is not a 2D square matrix.
        ValueError: If mat_x is not a positive semidefinite matrix.
        ValueError: If a constant CVXPY expression has no numeric ``.value``.
        ValueError: If affine or variable CVXPY inputs are passed.

    Returns:
        The quantum entropy of the matrix as a float.

    """
    if not isinstance(mat_x, (np.ndarray, cvxpy.Expression)):
        raise ValueError("mat_x must be a numpy array or a cvxpy expression")
    _require_square_2d(mat_x, "mat_x")

    if isinstance(mat_x, np.ndarray):
        if not is_positive_semidefinite(mat_x):
            raise ValueError("mat_x must be a positive semidefinite matrix")
        return float(np.real(-np.trace(mat_x @ logm(mat_x))))

    if isinstance(mat_x, cvxpy.Expression) and mat_x.is_constant():
        x_val = mat_x.value
        if x_val is None:
            raise ValueError(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass mat_x as a numpy.ndarray."
            )
        return ln_quantum_entropy(np.asarray(x_val))

    _reject_nonconstant_cvxpy(mat_x)
