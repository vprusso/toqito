"""Compute relative entropy quadrature from CVXQUAD."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np

from toqito.cones.operator_relative_entropy_epi_cone import (
    operator_relative_entropy_epi_cone,
)


def _is_constant(expr: np.ndarray | cvxpy.Expression) -> bool:
    if isinstance(expr, np.ndarray):
        return True
    return expr.is_constant()


def _constant_value(expr: np.ndarray | cvxpy.Expression) -> np.ndarray:
    if isinstance(expr, np.ndarray):
        return np.asarray(expr, dtype=float)
    if expr.value is None:
        raise ValueError("Constant CVXPY expression has no numeric value; set `.value` or pass a numpy.ndarray.")
    return np.asarray(expr.value, dtype=float)


def _broadcast_shape(
    vec_x: np.ndarray | cvxpy.Expression,
    vec_y: np.ndarray | cvxpy.Expression,
) -> tuple[np.ndarray | cvxpy.Expression, np.ndarray | cvxpy.Expression, tuple[int, ...]]:
    """Broadcast ``vec_x`` and ``vec_y`` like CVXQUAD ``rel_entr_quad.m``."""
    x_size = int(np.size(vec_x))
    y_size = int(np.size(vec_y))

    if x_size == 1:
        sz = np.shape(vec_y)
        if isinstance(vec_x, np.ndarray):
            vec_x = np.broadcast_to(vec_x, sz)
    elif y_size == 1:
        sz = np.shape(vec_x)
        if isinstance(vec_y, np.ndarray):
            vec_y = np.broadcast_to(vec_y, sz)
    elif np.shape(vec_x) == np.shape(vec_y):
        sz = np.shape(vec_x)
    else:
        raise ValueError("The dimensions of vec_x and vec_y are not compatible.")

    return vec_x, vec_y, sz


def relative_entropy_quadrature(
    vec_x: np.ndarray | cvxpy.Expression,
    vec_y: np.ndarray | cvxpy.Expression,
    m: int = 3,
    k: int = 3,
) -> np.ndarray | float:
    r"""Compute element-wise relative entropy \(x_i \log(x_i / y_i)\) using quadrature SDPs.

    For numeric inputs this returns the element-wise values. For affine CVXPY inputs
    it builds the separable quadrature SDP from CVXQUAD: each \(z_i\) lies in the
    epigraph of \(x_i \log(x_i / y_i)\), and the problem minimizes
    \(\sum_i z_i\). The returned scalar is that optimal value
    [@fawzi2015matrixgeometric].

    Args:
        vec_x: Positive vector (or CVXPY expression).
        vec_y: Positive vector (or CVXPY expression), broadcastable with ``vec_x``.
        m: Number of quadrature nodes.
        k: Number of square-roots in the approximation.

    Returns:
        Element-wise values for numeric inputs, or the minimized sum of epigraph
        variables for affine CVXPY inputs.

    Raises:
        ValueError: If inputs are not numpy arrays or CVXPY expressions.
        ValueError: If ``m`` or ``k`` is less than 1.
        ValueError: If shapes are incompatible.
        ValueError: If inputs are not positive at evaluation points.
        ValueError: If affine CVXPY inputs are not affine or lack initial values.
        ValueError: If the SDP does not solve successfully.

    Examples:
        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.cones import relative_entropy_quadrature
        vec_x = np.array([0.3, 0.7])
        vec_y = np.array([0.5, 0.5])
        print(relative_entropy_quadrature(vec_x, vec_y))
        ```

    """
    if not isinstance(vec_x, (np.ndarray, cvxpy.Expression)):
        raise ValueError("vec_x must be a numpy array or a cvxpy expression")
    if not isinstance(vec_y, (np.ndarray, cvxpy.Expression)):
        raise ValueError("vec_y must be a numpy array or a cvxpy expression")
    if m < 1:
        raise ValueError("m must be at least 1")
    if k < 1:
        raise ValueError("k must be at least 1")

    vec_x, vec_y, sz = _broadcast_shape(vec_x, vec_y)

    if isinstance(vec_x, np.ndarray) and isinstance(vec_y, np.ndarray):
        x_b = np.asarray(vec_x, dtype=float)
        y_b = np.asarray(vec_y, dtype=float)
        if np.any(x_b <= 0) or np.any(y_b <= 0):
            raise ValueError("vec_x and vec_y must be positive")
        return x_b * np.log(x_b / y_b)

    if _is_constant(vec_x) and _is_constant(vec_y):
        return relative_entropy_quadrature(_constant_value(vec_x), _constant_value(vec_y), m, k)

    if isinstance(vec_x, np.ndarray):
        vec_x = cvxpy.Constant(vec_x)
    if isinstance(vec_y, np.ndarray):
        vec_y = cvxpy.Constant(vec_y)

    if not (vec_x.is_affine() or vec_y.is_affine()):
        raise ValueError("At least one of vec_x and vec_y must be an affine CVXPY expression.")
    if vec_x.value is None or vec_y.value is None:
        raise ValueError("Affine vec_x and vec_y need numeric initial values; set `.value`.")
    if np.any(np.asarray(vec_x.value).reshape(-1) <= 0) or np.any(np.asarray(vec_y.value).reshape(-1) <= 0):
        raise ValueError("vec_x and vec_y must be positive at the initial value.")

    n = int(np.prod(sz))
    z_var = cvxpy.Variable(sz)
    z_flat = cvxpy.reshape(z_var, (n,), order="C")
    x_scalar = int(vec_x.size) == 1
    y_scalar = int(vec_y.size) == 1
    x_flat = None if x_scalar else cvxpy.reshape(vec_x, (n,), order="C")
    y_flat = None if y_scalar else cvxpy.reshape(vec_y, (n,), order="C")

    constraints = []
    eye1 = np.eye(1)
    for idx in range(n):
        xi = cvxpy.reshape(vec_x, (1, 1), order="C") if x_scalar else cvxpy.reshape(x_flat[idx], (1, 1), order="C")
        yi = cvxpy.reshape(vec_y, (1, 1), order="C") if y_scalar else cvxpy.reshape(y_flat[idx], (1, 1), order="C")
        zi = cvxpy.reshape(z_flat[idx], (1, 1), order="C")
        constraints.extend(
            operator_relative_entropy_epi_cone(
                xi,
                yi,
                zi,
                m=m,
                k=k,
                e=eye1,
                apx=0,
                hermitian=False,
            )
        )

    prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum(z_var)), constraints)
    result = prob.solve(solver=cvxpy.SCS, verbose=False)
    if prob.status not in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE):
        raise ValueError(f"The SDP did not solve successfully (status: {prob.status}).")
    return float(result)
