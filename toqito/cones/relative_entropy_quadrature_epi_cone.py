"""CVXPY constraints for the epigraph of element-wise relative entropy."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np

from toqito.cones.operator_relative_entropy_epi_cone import (
    operator_relative_entropy_epi_cone,
)


def _expr_size(expr: cvxpy.Expression) -> int:
    if not expr.shape:
        return 1
    return int(np.prod(expr.shape))


def _broadcast_expressions(
    vec_x: cvxpy.Expression,
    vec_y: cvxpy.Expression,
) -> tuple[cvxpy.Expression, cvxpy.Expression, tuple[int, ...]]:
    """Broadcast ``vec_x`` and ``vec_y`` like CVXQUAD ``rel_entr_quad.m``."""
    x_size = _expr_size(vec_x)
    y_size = _expr_size(vec_y)

    if x_size == 1:
        sz = tuple(int(d) for d in vec_y.shape)
        if sz and x_size != y_size:
            vec_x = cvxpy.multiply(vec_x, np.ones(sz))
    elif y_size == 1:
        sz = tuple(int(d) for d in vec_x.shape)
        # ``x_size != 1`` here, so ``sz`` is nonempty for ordinary CVXPY shapes.
        vec_y = cvxpy.multiply(vec_y, np.ones(sz))
    elif vec_x.shape == vec_y.shape:
        sz = tuple(int(d) for d in vec_x.shape)
    else:
        raise ValueError("The dimensions of vec_x and vec_y are not compatible.")

    return vec_x, vec_y, sz


def relative_entropy_quadrature_epi_cone(
    vec_x: cvxpy.Expression,
    vec_y: cvxpy.Expression,
    z: cvxpy.Expression,
    *,
    m: int = 3,
    k: int = 3,
    apx: int = 0,
) -> list[cvxpy.Constraint]:
    r"""Return CVXPY constraints for the epigraph of element-wise relative entropy.

    For positive vectors (or arrays) \(x\) and \(y\), the map

    \[
        (x, y) \mapsto x \odot \log(x \oslash y)
    \]

    is jointly convex. The constraints enforce

    \[
        z \geqslant x \odot \log(x \oslash y)
    \]

    element-wise by wrapping ``operator_relative_entropy_epi_cone`` on each
    scalar as a ``1 x 1`` block, matching CVXQUAD ``rel_entr_quad.m``
    [@fawzi2015matrixgeometric].

    Args:
        vec_x: A CVXPY expression for a positive vector / array.
        vec_y: A CVXPY expression for a positive vector / array, broadcastable
            with ``vec_x``.
        z: A CVXPY expression for the epigraph variable, with the broadcast
            shape of ``vec_x`` and ``vec_y``.
        m: The number of quadrature nodes to use.
        k: The number of square-roots to take.
        apx: Approximation mode passed to
            ``operator_relative_entropy_epi_cone``: ``-1`` / ``0`` / ``1``.

    Raises:
        ValueError: If ``vec_x`` and ``vec_y`` have incompatible shapes.
        ValueError: If ``z`` does not match the broadcast shape.
        ValueError: If ``m`` or ``k`` is less than 1.
        ValueError: If ``apx`` is not ``-1``, ``0``, or ``1``.

    Returns:
        A list of CVXPY constraints.

    Examples:
        ```python exec="1" source="above" result="text"
        import cvxpy
        import numpy as np
        from toqito.cones import relative_entropy_quadrature_epi_cone

        vec_x = np.array([0.3, 0.7])
        vec_y = np.array([0.5, 0.5])
        z = cvxpy.Variable(2)
        cons = relative_entropy_quadrature_epi_cone(
            cvxpy.Constant(vec_x),
            cvxpy.Constant(vec_y),
            z,
        )
        prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum(z)), cons)
        print(f"{prob.solve(solver=cvxpy.SCS, verbose=False):.4f}")
        ```

    """
    if m < 1:
        raise ValueError("m must be at least 1")
    if k < 1:
        raise ValueError("k must be at least 1")
    if apx not in (-1, 0, 1):
        raise ValueError("apx must be -1, 0, or 1")

    vec_x, vec_y, sz = _broadcast_expressions(vec_x, vec_y)
    z_shape = tuple(int(d) for d in z.shape)
    if z_shape != sz:
        raise ValueError("z must have the broadcast shape of vec_x and vec_y")

    n = int(np.prod(sz)) if sz else 1
    if n == 0:
        return []

    x_flat = cvxpy.reshape(vec_x, (n,), order="F")
    y_flat = cvxpy.reshape(vec_y, (n,), order="F")
    z_flat = cvxpy.reshape(z, (n,), order="F")

    constraints: list[cvxpy.Constraint] = []
    for i in range(n):
        x_i = cvxpy.reshape(x_flat[i], (1, 1), order="F")
        y_i = cvxpy.reshape(y_flat[i], (1, 1), order="F")
        z_i = cvxpy.reshape(z_flat[i], (1, 1), order="F")
        constraints.extend(
            operator_relative_entropy_epi_cone(
                x_i,
                y_i,
                z_i,
                m=m,
                k=k,
                apx=apx,
                hermitian=False,
            )
        )
    return constraints
