"""Compute relative entropy quadrature from CVXQUAD."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np

from toqito.cones._utils import _reject_nonconstant_cvxpy


def _constant_value(expr: cvxpy.Expression) -> np.ndarray:
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

    For numeric inputs this returns the element-wise values
    [@fawzi2015matrixgeometric]. Constant CVXPY expressions with a concrete
    ``.value`` are routed through the numeric path. Affine or variable CVXPY
    inputs are not yet supported.

    Args:
        vec_x: Positive vector (or CVXPY expression).
        vec_y: Positive vector (or CVXPY expression), broadcastable with ``vec_x``.
        m: Number of quadrature nodes (reserved for a future affine implementation).
        k: Number of square-roots in the approximation (reserved for a future affine implementation).

    Returns:
        Element-wise values for numeric inputs.

    Raises:
        ValueError: If inputs are not numpy arrays or CVXPY expressions.
        ValueError: If ``m`` or ``k`` is less than 1.
        ValueError: If shapes are incompatible.
        ValueError: If inputs are not positive at evaluation points.
        ValueError: If a constant CVXPY expression has no numeric ``.value``.
        ValueError: If affine or variable CVXPY inputs are passed.

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

    vec_x, vec_y, _sz = _broadcast_shape(vec_x, vec_y)

    if isinstance(vec_x, np.ndarray) and isinstance(vec_y, np.ndarray):
        x_b = np.asarray(vec_x, dtype=float)
        y_b = np.asarray(vec_y, dtype=float)
        if np.any(x_b <= 0) or np.any(y_b <= 0):
            raise ValueError("vec_x and vec_y must be positive")
        return x_b * np.log(x_b / y_b)

    if isinstance(vec_x, np.ndarray):
        vec_x = cvxpy.Constant(vec_x)
    if isinstance(vec_y, np.ndarray):
        vec_y = cvxpy.Constant(vec_y)

    if vec_x.is_constant() and vec_y.is_constant():
        return relative_entropy_quadrature(
            _constant_value(vec_x),
            _constant_value(vec_y),
            m,
            k,
        )

    _reject_nonconstant_cvxpy(vec_x, vec_y)
