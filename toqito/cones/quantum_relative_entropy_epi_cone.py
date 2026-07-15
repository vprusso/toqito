"""CVXPY constraints for the quantum relative entropy epigraph cone."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np

from toqito.cones._utils import _require_square_2d, _symmetric_like_variable
from toqito.cones.operator_relative_entropy_epi_cone import (
    operator_relative_entropy_epi_cone,
)


def quantum_relative_entropy_epi_cone(
    mat_x: cvxpy.Expression,
    mat_y: cvxpy.Expression,
    t: cvxpy.Expression,
    *,
    m: int = 3,
    k: int = 3,
    apx: int = 0,
    hermitian: bool = False,
) -> list[cvxpy.Constraint]:
    r"""Return CVXPY constraints for the epigraph of quantum relative entropy.

    The quantum relative entropy \(D(X\|Y) = \operatorname{tr}(X\log X - X\log Y)\)
    (natural log) is jointly convex in PSD \((X, Y)\). The constraints enforce

    \[
        t \geqslant D(X\|Y)
    \]

    via the Kronecker lifting used in CVXQUAD ``quantum_rel_entr.m``
    [@fawzi2017matrixlogarithm]: an auxiliary scalar matrix ``TAU`` satisfies

    \[
        D_{\mathrm{op}}\!\bigl(X \otimes I,\; I \otimes \overline{Y}\bigr)
        \preceq \mathrm{TAU},
        \qquad
        t \geqslant \operatorname{Re}(\mathrm{TAU}),
    \]

    with the vectorization weight \(e = \operatorname{vec}(I)\).

    Args:
        mat_x: A CVXPY expression for an ``n x n`` PSD matrix.
        mat_y: A CVXPY expression for an ``n x n`` PSD matrix.
        t: A CVXPY scalar (or ``1 x 1``) epigraph variable.
        m: The number of quadrature nodes to use.
        k: The number of square-roots to take.
        apx: Approximation mode passed to
            ``operator_relative_entropy_epi_cone``: ``-1`` / ``0`` / ``1``.
        hermitian: Whether the matrices are Hermitian or symmetric.

    Raises:
        ValueError: If ``mat_x`` or ``mat_y`` is not square 2D.
        ValueError: If ``mat_x`` and ``mat_y`` do not have the same shape.
        ValueError: If ``m`` or ``k`` is less than 1.
        ValueError: If ``apx`` is not ``-1``, ``0``, or ``1``.

    Returns:
        A list of CVXPY constraints.

    Examples:
        ```python exec="1" source="above" result="text"
        import cvxpy
        import numpy as np
        from toqito.cones import quantum_relative_entropy_epi_cone

        rho = np.array([[1.0, 0.0], [0.0, 0.0]])
        sigma = np.eye(2) / 2
        t = cvxpy.Variable()
        cons = quantum_relative_entropy_epi_cone(
            cvxpy.Constant(rho),
            cvxpy.Constant(sigma),
            t,
        )
        prob = cvxpy.Problem(cvxpy.Minimize(t), cons)
        print(f"{prob.solve(solver=cvxpy.SCS, verbose=False):.4f}")
        ```

    """
    _require_square_2d(mat_x, "mat_x")
    _require_square_2d(mat_y, "mat_y")
    if mat_x.shape != mat_y.shape:
        raise ValueError("mat_x and mat_y must have the same shape")
    if m < 1:
        raise ValueError("m must be at least 1")
    if k < 1:
        raise ValueError("k must be at least 1")
    if apx not in (-1, 0, 1):
        raise ValueError("apx must be -1, 0, or 1")

    n = int(mat_x.shape[0])
    tau = _symmetric_like_variable(1, hermitian=hermitian)
    e = np.reshape(np.eye(n), (-1, 1), order="F")
    mat_x_kron = cvxpy.kron(mat_x, np.eye(n))
    mat_y_kron = cvxpy.kron(np.eye(n), cvxpy.conj(mat_y))
    constraints = operator_relative_entropy_epi_cone(
        mat_x_kron,
        mat_y_kron,
        tau,
        m=m,
        k=k,
        e=e,
        apx=apx,
        hermitian=hermitian,
    )
    bound = tau
    if hermitian:
        bound = cvxpy.real(bound)
    constraints.append(t >= bound)
    return constraints
