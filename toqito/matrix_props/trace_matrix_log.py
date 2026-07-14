r"""Computes \(\operatorname{tr}(C \log A)\) for positive semidefinite matrices \(A\) and \(C\)."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np
from scipy.linalg import logm

from toqito.cones._utils import _reject_nonconstant_cvxpy, _require_square_2d
from toqito.matrix_props import is_positive_semidefinite


def trace_matrix_log(
    mat_x: np.ndarray | cvxpy.Expression,
    mat_c: np.ndarray | None = None,
) -> float:
    r"""Return \(\operatorname{tr}(C \log A)\) for positive semidefinite matrices \(A\) and \(C\).

    If `mat_c` is not provided, it is set to the identity matrix.

    Assuming that `mat_c` is a fixed positive semidefinite matrix, the function
    is concave in `mat_x` [@fawzi2017matrixlogarithm].

    This function evaluates the formula numerically. Constant CVXPY expressions
    with a concrete ``.value`` are routed through the numeric path. Affine or
    variable CVXPY inputs are not supported; use
    :func:`~toqito.cones.trace_matrix_log_hypo_cone` for composition in a
    parent SDP (with quadrature parameters ``m``, ``k``, ``apx``).

    Args:
        mat_x: A numpy array or constant CVXPY expression representing a positive
            semidefinite matrix.
        mat_c: A numpy array representing a positive semidefinite matrix.

    Raises:
        ValueError: If mat_x is not a square 2D matrix.
        ValueError: If mat_c is not a square 2D matrix.
        ValueError: If mat_x is not a numpy array or a cvxpy expression.
        ValueError: If mat_x is not a positive semidefinite matrix.
        ValueError: If mat_c is not a numpy array.
        ValueError: If mat_c is not a positive semidefinite matrix.
        ValueError: If mat_x and mat_c do not have the same shape.
        ValueError: If a constant CVXPY expression has no numeric ``.value``.
        ValueError: If affine or variable CVXPY inputs are passed.

    Returns:
        A float representing the value of the trace of the matrix logarithm.

    """
    if not isinstance(mat_x, (np.ndarray, cvxpy.Expression)):
        raise ValueError("mat_x must be a numpy array or a cvxpy expression")
    _require_square_2d(mat_x, "mat_x")
    if mat_c is None:
        mat_c = np.eye(mat_x.shape[0])
    else:
        if not isinstance(mat_c, np.ndarray):
            raise ValueError("mat_c must be a numpy array")
        _require_square_2d(mat_c, "mat_c")
        if not is_positive_semidefinite(mat_c):
            raise ValueError("mat_c must be a positive semidefinite matrix")
    if mat_x.shape != mat_c.shape:
        raise ValueError("mat_x and mat_c must have the same shape")

    if isinstance(mat_x, np.ndarray):
        if not is_positive_semidefinite(mat_x):
            raise ValueError("mat_x must be a positive semidefinite matrix")
        return float(np.real(np.trace(mat_c @ logm(mat_x))))

    if isinstance(mat_x, cvxpy.Expression) and mat_x.is_constant():
        x_val = mat_x.value
        if x_val is None:
            raise ValueError(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass mat_x as a numpy.ndarray."
            )
        return trace_matrix_log(np.asarray(x_val), mat_c)

    _reject_nonconstant_cvxpy(mat_x)
