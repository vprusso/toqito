r"""Tsallis relative entropy for positive semidefinite matrices."""

import cvxpy
import numpy as np

from toqito.cones._utils import _reject_nonconstant_cvxpy, _require_square_2d
from toqito.matrix_props import is_positive_semidefinite
from toqito.matrix_props.lieb_ando import lieb_ando


def tsallis_relative_entropy(
    mat_x: np.ndarray | cvxpy.Expression,
    mat_y: np.ndarray | cvxpy.Expression,
    t: float,
) -> float:
    r"""Compute the Tsallis relative entropy \(S_t(A\|B)\) [@fawzi2015matrixgeometric].

    For \(t \in [0, 1]\), the Tsallis relative entropy is defined by

    \[
        S_t(A\|B) = \frac{1}{t}\operatorname{tr}\!\bigl(A - A^{1-t} B^t\bigr).
    \]

    As \(t \to 0^+\), this converges to the quantum relative entropy

    \[
        D(A\|B) = \operatorname{tr}\!\bigl[A(\log A - \log B)\bigr],
    \]

    with convergence from below, i.e. \(S_t(A\|B) \leqslant D(A\|B)\) for all
    \(t \in [0, 1]\). The map \((A, B) \mapsto S_t(A\|B)\) is jointly convex on
    \(\text{H}_n^{++} \times \text{H}_n^{++}\).

    This function evaluates the formula numerically. Constant CVXPY expressions
    with a concrete ``.value`` are routed through the numeric path. Affine or
    variable CVXPY inputs are not yet supported.

    Args:
        mat_x: The first positive semidefinite matrix \(A\), or a constant CVXPY expression.
        mat_y: The second positive semidefinite matrix \(B\), or a constant CVXPY expression.
        t: Order parameter in the range ``[0, 1]``.

    Raises:
        ValueError: If ``mat_x`` or ``mat_y`` is not a numpy array or a cvxpy expression.
        ValueError: If ``mat_x`` or ``mat_y`` is not a 2D square matrix.
        ValueError: If ``mat_x`` and ``mat_y`` do not have the same shape.
        ValueError: If ``t`` is not in the range ``[0, 1]``.
        ValueError: If ``mat_x`` or ``mat_y`` is not positive semidefinite.
        ValueError: If a constant CVXPY expression has no numeric ``.value``.
        ValueError: If affine or variable CVXPY inputs are passed.

    Returns:
        The Tsallis relative entropy \(S_t(A\|B)\) as a float.

    Examples:
        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_props import tsallis_relative_entropy
        mat_x = np.diag([0.25, 0.75])
        mat_y = np.diag([0.5, 0.5])
        t = 0.5
        print(tsallis_relative_entropy(mat_x, mat_y, t))
        ```

    """
    if not isinstance(mat_x, (np.ndarray, cvxpy.Expression)):
        raise ValueError("mat_x must be a numpy array or a cvxpy expression")
    if not isinstance(mat_y, (np.ndarray, cvxpy.Expression)):
        raise ValueError("mat_y must be a numpy array or a cvxpy expression")
    _require_square_2d(mat_x, "mat_x")
    _require_square_2d(mat_y, "mat_y")
    if mat_x.shape != mat_y.shape:
        raise ValueError("mat_x and mat_y must have the same shape")
    if t < 0 or t > 1:
        raise ValueError("t must be in the range [0, 1]")

    if isinstance(mat_x, np.ndarray) and isinstance(mat_y, np.ndarray):
        if not is_positive_semidefinite(mat_x):
            raise ValueError("mat_x must be a positive semidefinite matrix")
        if not is_positive_semidefinite(mat_y):
            raise ValueError("mat_y must be a positive semidefinite matrix")
        if t == 0:
            from toqito.state_props.quantum_relative_entropy import (
                quantum_relative_entropy,
            )  # noqa: PLC0415

            return quantum_relative_entropy(mat_x, mat_y)
        n = int(mat_x.shape[0])
        trace_cross = lieb_ando(mat_x, mat_y, np.eye(n), t)
        return float(np.real((np.trace(mat_x) - trace_cross) / t))

    if isinstance(mat_x, np.ndarray):
        mat_x = cvxpy.Constant(mat_x)
    if isinstance(mat_y, np.ndarray):
        mat_y = cvxpy.Constant(mat_y)

    if mat_x.is_constant() and mat_y.is_constant():
        if mat_x.value is None or mat_y.value is None:
            raise ValueError(
                "Constant CVXPY expression has no numeric value; set parameter `.value` or pass a numpy.ndarray."
            )
        return tsallis_relative_entropy(
            np.asarray(mat_x.value), np.asarray(mat_y.value), t
        )

    _reject_nonconstant_cvxpy(mat_x, mat_y)
