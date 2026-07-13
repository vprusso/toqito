"""CVXPY constraints for the hypograph of the matrix logarithm trace."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np

from toqito.cones._utils import _require_square_2d, _symmetric_like_variable
from toqito.cones.operator_relative_entropy_epi_cone import (
    operator_relative_entropy_epi_cone,
)
from toqito.matrix_props import is_positive_semidefinite


def trace_matrix_log_hypo_cone(
    mat_x: cvxpy.Expression,
    t: cvxpy.Expression,
    mat_c: np.ndarray | None = None,
    *,
    m: int = 3,
    k: int = 3,
    apx: int = 0,
    hermitian: bool = False,
) -> list[cvxpy.Constraint]:
    r"""Return CVXPY constraints for the hypograph of \(\operatorname{tr}(C \log X)\).

    The constraints enforce

    \[
        t \leqslant \operatorname{tr}(C \log X)
    \]

    approximately via the operator relative entropy identity
    \(\operatorname{tr}(C \log X) = -\operatorname{tr}\!\bigl(C\, D_{\mathrm{op}}(I\|X)\bigr)\).
    An auxiliary matrix ``TAU`` is introduced internally with

    \[
        D_{\mathrm{op}}(I\|X) \preceq \mathrm{TAU},
        \qquad
        t \leqslant -\operatorname{tr}(C\,\mathrm{TAU}).
    \]

    If ``mat_c`` is omitted, it defaults to the identity. The ``apx`` sign is
    flipped when calling ``operator_relative_entropy_epi_cone``
    so that upper/lower bounds match CVXQUAD ``trace_logm.m``
    [@fawzi2017matrixlogarithm].

    Args:
        mat_x: A CVXPY expression for a positive semidefinite matrix.
        t: A CVXPY scalar (or ``1 x 1``) hypograph variable.
        mat_c: A fixed positive semidefinite weight matrix (default: identity).
        m: The number of quadrature nodes to use.
        k: The number of square-roots to take.
        apx: Approximation mode for the log-trace bound: ``+1`` upper-bounds
            \(\operatorname{tr}(C \log X)\), ``-1`` lower-bounds it, and ``0``
            is the two-sided Padé / Gauss--Legendre approximation.
        hermitian: Whether the matrices are Hermitian or symmetric.

    Raises:
        ValueError: If ``mat_x`` is not a square 2D expression.
        ValueError: If ``mat_c`` is not a square PSD numpy array of matching shape.
        ValueError: If ``m`` or ``k`` is less than 1.
        ValueError: If ``apx`` is not ``-1``, ``0``, or ``1``.

    Returns:
        A list of CVXPY constraints.

    Examples:
        ```python exec="1" source="above" result="text"
        import cvxpy
        import numpy as np
        from toqito.cones import trace_matrix_log_hypo_cone

        n = 2
        mat_x = np.eye(n) / n
        x_c = cvxpy.Constant(mat_x)
        t = cvxpy.Variable()
        cons = trace_matrix_log_hypo_cone(x_c, t, m=3, k=3, apx=0)
        prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
        print(f"{prob.solve(solver=cvxpy.SCS, verbose=False):.4f}")
        ```

    """
    _require_square_2d(mat_x, "mat_x")
    n = int(mat_x.shape[0])
    if mat_c is None:
        mat_c = np.eye(n)
    else:
        if not isinstance(mat_c, np.ndarray):
            raise ValueError("mat_c must be a numpy array")
        _require_square_2d(mat_c, "mat_c")
        if not is_positive_semidefinite(mat_c):
            raise ValueError("mat_c must be a positive semidefinite matrix")
    if mat_x.shape != mat_c.shape:
        raise ValueError("mat_x and mat_c must have the same shape")
    if m < 1:
        raise ValueError("m must be at least 1")
    if k < 1:
        raise ValueError("k must be at least 1")
    if apx not in (-1, 0, 1):
        raise ValueError("apx must be -1, 0, or 1")

    tau = _symmetric_like_variable(n, hermitian=hermitian)
    eye_n = cvxpy.Constant(np.eye(n))

    constraints = operator_relative_entropy_epi_cone(
        eye_n,
        mat_x,
        tau,
        m=m,
        k=k,
        e=np.eye(n),
        apx=-apx,
        hermitian=hermitian,
    )
    c_expr = cvxpy.Constant(mat_c)
    weighted_trace = cvxpy.trace(c_expr @ tau)
    if hermitian:
        weighted_trace = cvxpy.real(weighted_trace)
    constraints.append(t <= -weighted_trace)
    return constraints
