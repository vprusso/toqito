r"""Computes the trace of \(C A^{t}\) for positive semidefinite matrices \(A\) and \(C\)."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np

from toqito.cones._utils import _reject_nonconstant_cvxpy, _require_square_2d
from toqito.matrix_ops import psd_matrix_power
from toqito.matrix_props import is_positive_semidefinite


def trace_matrix_power(mat_a: np.ndarray | cvxpy.Expression, t: float, mat_c: np.ndarray | None = None) -> float:
    r"""Return \(\operatorname{tr}\!\bigl(C A^{t}\bigr)\) for PSD matrices ``mat_a`` and ``mat_c``.

    Here \(A=\) ``mat_a`` and \(C=\) ``mat_c`` [@fawzi2015matrixgeometric].

    ``mat_c`` is optional. If not provided, it is assumed to be the identity matrix.
    When `t` is in the range `[0, 1]`, the function is concave in ``mat_a`` for
    fixed ``mat_c``. When `t` is in the range `[-1, 0]` or `[1, 2]`,
    the function is convex in ``mat_a`` for fixed ``mat_c``.

    This function evaluates the formula numerically. Constant CVXPY expressions
    with a concrete ``.value`` are routed through the numeric path. Affine or
    variable CVXPY inputs are not supported; use
    ``trace_matrix_power_hypo_cone`` (for ``t`` in ``[0, 1]``) or
    ``trace_matrix_power_epi_cone`` (for ``t`` in ``[-1, 0]`` or ``[1, 2]``)
    for composition in a parent SDP.

    Args:
        mat_a: A numpy array or constant CVXPY expression representing a positive
            semidefinite matrix.
        t: The power to raise the matrix to.
        mat_c: A numpy array representing a positive semidefinite weight matrix.

    Returns:
        \(\operatorname{tr}\!\bigl(C A^{t}\bigr)\) with \(A=\) ``mat_a`` and \(C=\) ``mat_c``.
        When ``mat_c`` is omitted, \(C = I\).

    Raises:
        TypeError: If `mat_a` is not a numpy.ndarray or a cvxpy expression.
        ValueError: If `mat_a` is not 2D or not square.
        TypeError: If `mat_c` is not a numpy.ndarray or None.
        ValueError: If `mat_c` is not 2D or not square (unless it is None).
        ValueError: If `mat_c` is not positive semidefinite (unless it is None).
        ValueError: If `mat_a` is not positive semidefinite.
        ValueError: If `mat_c` is not the same size as `mat_a` (unless it is None).
        ValueError: If a constant CVXPY expression has no numeric ``.value``.
        ValueError: If affine or variable CVXPY inputs are passed.

    Examples:
        ```python
        import numpy as np
        from toqito.matrix_props import trace_matrix_power
        mat_a = np.array([[2.0, 1.0], [1.0, 2.0]])
        t = 0.5
        mat_c = np.array([[1, 0], [0, 1]])
        print(trace_matrix_power(mat_a, t, mat_c))
        ```

    """
    if not isinstance(mat_a, (np.ndarray, cvxpy.Expression)):
        raise TypeError("mat_a must be a numpy.ndarray or a cvxpy expression.")

    _require_square_2d(mat_a, "mat_a")

    if mat_c is not None and not isinstance(mat_c, np.ndarray):
        raise TypeError("mat_c must be a numpy.ndarray or None.")

    if mat_c is None:
        mat_c = np.eye(mat_a.shape[0])

    _require_square_2d(mat_c, "mat_c")
    if not is_positive_semidefinite(mat_c):
        raise ValueError("The matrix mat_c must be positive semidefinite.")

    if mat_a.shape != mat_c.shape:
        raise ValueError("The matrices must be the same size.")

    if isinstance(mat_a, np.ndarray):
        if not is_positive_semidefinite(mat_a):
            raise ValueError("The matrix mat_a must be positive semidefinite.")
        return float(np.real(np.trace(mat_c @ psd_matrix_power(mat_a, t))))

    if isinstance(mat_a, cvxpy.Expression) and mat_a.is_constant():
        a_val = mat_a.value
        if a_val is None:
            raise ValueError(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass mat_a as a numpy.ndarray."
            )
        return trace_matrix_power(np.asarray(a_val), t, mat_c)

    _reject_nonconstant_cvxpy(mat_a)
