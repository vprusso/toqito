r"""Computes \(\operatorname{tr}(C \log A)\) for positive semidefinite matrices \(A\) and \(C\)."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np
from scipy.linalg import logm

from toqito.cones._utils import _AFFINE_VARIABLE_NOT_SUPPORTED, _require_square_2d
from toqito.matrix_props import is_positive_semidefinite


def trace_matrix_log(
    mat_x: np.ndarray | cvxpy.Expression,
    mat_c: np.ndarray | None = None,
    m: int = 3,
    k: int = 3,
    apx: int = 0,
) -> float:
    r"""Return \(\operatorname{tr}(C \log A)\) for positive semidefinite matrices \(A\) and \(C\).

    If `mat_c` is not provided, it is set to the identity matrix.

    Assuming that `mat_c` is a fixed positive semidefinite matrix, the function
    is concave in `mat_x` [@fawzi2017matrixlogarithm].

    The parameters `m` and `k` control quadrature accuracy; `apx` sets lower-bound,
    two-sided (Gauss--Legendre), or upper-bound quadrature for the matrix logarithm.

    Args:
        mat_x: A numpy array or a cvxpy expression representing a positive semidefinite matrix.
        mat_c: A numpy array representing a positive semidefinite matrix.
        m: The number of quadrature nodes to use.
        k: The number of square-roots to take.
        apx: Same as in ``operator_relative_entropy_epi_cone``: ``-1`` lower bound
            (left Gauss--Radau), ``0`` two-sided Gauss--Legendre, ``1`` upper bound
            (right Gauss--Radau).

    Raises:
        ValueError: If mat_x is not a square 2D matrix.
        ValueError: If mat_c is not a square 2D matrix.
        ValueError: If mat_x is not a numpy array or a cvxpy expression.
        ValueError: If mat_x is not a positive semidefinite matrix.
        ValueError: If mat_c is not a numpy array.
        ValueError: If mat_c is not a positive semidefinite matrix.
        ValueError: If mat_x and mat_c do not have the same shape.
        ValueError: If mat_x is not an affine CVXPY expression.
        ValueError: If mat_x has no numeric initial value.


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
        return trace_matrix_log(np.asarray(x_val), mat_c, m, k, apx)

    # Affine non-constant expression: evaluate numerically via .value.
    if not mat_x.is_affine():
        raise ValueError("mat_x must be an affine CVXPY expression.")
    if mat_x.value is None:
        raise ValueError(_AFFINE_VARIABLE_NOT_SUPPORTED)
    x_val = np.asarray(mat_x.value)
    if not is_positive_semidefinite(x_val):
        raise ValueError(_AFFINE_VARIABLE_NOT_SUPPORTED)
    return trace_matrix_log(x_val, mat_c, m, k, apx)
