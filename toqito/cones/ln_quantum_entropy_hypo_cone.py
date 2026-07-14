"""CVXPY constraints for the von Neumann entropy hypograph cone."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np

from toqito.cones._utils import _require_square_2d, _symmetric_like_variable
from toqito.cones.operator_relative_entropy_epi_cone import (
    operator_relative_entropy_epi_cone,
)


def ln_quantum_entropy_hypo_cone(
    mat_x: cvxpy.Expression,
    t: cvxpy.Expression,
    *,
    m: int = 3,
    k: int = 3,
    apx: int = 0,
    hermitian: bool = False,
) -> list[cvxpy.Constraint]:
    r"""Return CVXPY constraints for the hypograph of \(H(X) = -\operatorname{tr}(X \log X)\).

    The constraints enforce

    \[
        t \leqslant H(X)
    \]

    approximately via the operator relative entropy identity
    \(H(X) = -\operatorname{tr} D_{\mathrm{op}}(X\|I)\). An auxiliary matrix
    ``TAU`` is introduced internally with

    \[
        D_{\mathrm{op}}(X\|I) \preceq \mathrm{TAU},
        \qquad
        t \leqslant -\operatorname{tr}(\mathrm{TAU}).
    \]

    The ``apx`` sign is flipped when calling
    ``operator_relative_entropy_epi_cone`` so that upper/lower
    bounds on entropy match CVXQUAD ``quantum_entr.m`` [@fawzi2017matrixlogarithm].

    Args:
        mat_x: A CVXPY expression for a positive semidefinite matrix.
        t: A CVXPY scalar (or ``1 x 1``) hypograph variable.
        m: The number of quadrature nodes to use.
        k: The number of square-roots to take.
        apx: Approximation mode for the entropy bound: ``+1`` upper-bounds
            \(H(X)\), ``-1`` lower-bounds \(H(X)\), and ``0`` is the two-sided
            Padé / Gauss--Legendre approximation.
        hermitian: Whether the matrices are Hermitian or symmetric.

    Raises:
        ValueError: If ``mat_x`` is not a square 2D expression.
        ValueError: If ``m`` or ``k`` is less than 1.
        ValueError: If ``apx`` is not ``-1``, ``0``, or ``1``.

    Returns:
        A list of CVXPY constraints.

    Examples:
        ```python exec="1" source="above" result="text"
        import cvxpy
        import numpy as np
        from toqito.cones import ln_quantum_entropy_hypo_cone

        n = 2
        mat_x = np.eye(n) / n
        x_c = cvxpy.Constant(mat_x)
        t = cvxpy.Variable()
        cons = ln_quantum_entropy_hypo_cone(x_c, t, m=3, k=3, apx=0)
        prob = cvxpy.Problem(cvxpy.Maximize(t), cons)
        print(f"{prob.solve(solver=cvxpy.SCS, verbose=False):.4f}")
        ```

    """
    _require_square_2d(mat_x, "mat_x")
    if m < 1:
        raise ValueError("m must be at least 1")
    if k < 1:
        raise ValueError("k must be at least 1")
    if apx not in (-1, 0, 1):
        raise ValueError("apx must be -1, 0, or 1")

    n = int(mat_x.shape[0])
    tau = _symmetric_like_variable(n, hermitian=hermitian)
    eye_n = cvxpy.Constant(np.eye(n))

    constraints = operator_relative_entropy_epi_cone(
        mat_x,
        eye_n,
        tau,
        m=m,
        k=k,
        e=np.eye(n),
        apx=-apx,
        hermitian=hermitian,
    )
    trace_tau = cvxpy.trace(tau)
    if hermitian:
        trace_tau = cvxpy.real(trace_tau)
    constraints.append(t <= -trace_tau)
    return constraints
