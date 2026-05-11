r"""Computes the trace of \(C A^{t}\) for positive semidefinite matrices \(A\) and \(C\)."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np
from scipy.linalg import fractional_matrix_power

from toqito.matrix_ops.geometric_mean_epi_cone import geometric_mean_epi_cone
from toqito.matrix_ops.geometric_mean_hypo_cone import geometric_mean_hypo_cone
from toqito.matrix_props import is_positive_semidefinite


def trace_power(
    mat_a: np.ndarray | cvxpy.Expression, t: float, mat_c: np.ndarray | None = None
) -> float:
    r"""Return \(\operatorname{tr}\!\bigl(C A^{t}\bigr)\) for PSD matrices ``mat_a`` and ``mat_c``.

    Here \(A=\) ``mat_a`` and \(C=\) ``mat_c`` [@fawzi2015matrixgeometric].

    ``mat_c`` is optional. If not provided, it is assumed to be the identity matrix.
    When `t` is in the range `[0, 1]`, the function is concave in ``mat_a`` for
    fixed ``mat_c``. When `t` is in the range `[-1, 0]` or `[1, 2]`,
    the function is convex in ``mat_a`` for fixed ``mat_c``.

    Args:
        mat_a: The matrix to be raised to a power.
        t: The power to raise the matrix to. If mat_a is a cvxpy expression, t must be in the range `[-1, 2]`.
        mat_c: The matrix to multiply the result by.

    Returns:
        \(\operatorname{tr}\!\bigl(C A^{t}\bigr)\) with \(A=\) ``mat_a`` and \(C=\) ``mat_c``.
        When ``mat_c`` is omitted, \(C = I\).

    Raises:
        ValueError: If `mat_a` is not 2D or not square.
        TypeError: If `mat_c` is not a numpy.ndarray or None.
        ValueError: If `mat_c` is not 2D or not square (unless it is None).
        ValueError: If `mat_c` is not positive semidefinite (unless it is None).
        ValueError: If `mat_a` is not positive semidefinite.
        ValueError: If `t` is not in the range `[-1, 2]` and mat_a is a cvxpy expression.
        ValueError: If `mat_a` is not an affine expression (unless it is a numpy array).
        ValueError: If `mat_c` is not the same size as `mat_a` (unless it is None).

    Examples:
        ```python
        import numpy as np
        from toqito.matrix_ops import trace_power
        mat_a = np.array([[1, 2], [3, 4]])
        t = 0.5
        mat_c = np.array([[1, 0], [0, 1]])
        print(trace_power(mat_a, t, mat_c))
        ```

    """
    if mat_a.ndim != 2:
        raise ValueError("mat_a must be 2D.")
    if mat_a.shape[0] != mat_a.shape[1]:
        raise ValueError("mat_a must be square.")

    if mat_c is not None and not isinstance(mat_c, np.ndarray):
        raise TypeError("mat_c must be a numpy.ndarray or None.")

    if mat_c is None:
        mat_c = np.eye(mat_a.shape[0])

    if mat_c.ndim != 2:
        raise ValueError("mat_c must be 2D.")
    if mat_c.shape[0] != mat_c.shape[1]:
        raise ValueError("mat_c must be square.")
    if not is_positive_semidefinite(mat_c):
        raise ValueError("The matrix mat_c must be positive semidefinite.")

    if mat_a.shape != mat_c.shape:
        raise ValueError("The matrices must be the same size.")

    # Numeric path
    if isinstance(mat_a, np.ndarray):
        if not is_positive_semidefinite(mat_a):
            raise ValueError("The matrix mat_a must be positive semidefinite.")
        return float(np.real(np.trace(mat_c @ fractional_matrix_power(mat_a, t))))

    # Affine path
    if isinstance(mat_a, cvxpy.Expression):
        if t < -1 or t > 2:
            raise ValueError("The exponent t must be in the range [-1, 2].")
        if not mat_a.is_affine():
            raise ValueError("The matrix mat_a must be an affine expression.")
        if not is_positive_semidefinite(mat_a.value):
            raise ValueError("The matrix mat_a must be positive semidefinite.")
        n = mat_a.shape[0]
        is_cplx = np.any(np.imag(mat_a.value) != 0)
        if is_cplx:
            T = cvxpy.Variable((n, n), hermitian=True)
        else:
            T = cvxpy.Variable((n, n), symmetric=True)

        c_expr = cvxpy.Constant(mat_c)

        obj = cvxpy.trace(c_expr @ T)
        if is_cplx:
            obj = cvxpy.real(obj)
        if t >= 0 and t <= 1:
            cons = geometric_mean_hypo_cone(
                cvxpy.Constant(np.eye(n)), mat_a, T, t, fullhyp=False, hermitian=is_cplx
            )
            problem = cvxpy.Problem(cvxpy.Maximize(obj), cons)
            return problem.solve()
        cons = geometric_mean_epi_cone(
            cvxpy.Constant(np.eye(n)), mat_a, T, t, hermitian=is_cplx
        )
        problem = cvxpy.Problem(cvxpy.Minimize(obj), cons)
        return problem.solve()

    raise ValueError("The matrix mat_a must be a numpy array or a cvxpy expression.")
