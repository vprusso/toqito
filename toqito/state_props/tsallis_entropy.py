"""Tsallis entropy for positive semidefinite matrices."""

import cvxpy
import numpy as np
from scipy.linalg import logm

from toqito.cones._utils import _reject_nonconstant_cvxpy, _require_square_2d
from toqito.matrix_ops.psd_matrix_power import psd_matrix_power
from toqito.matrix_props import is_positive_semidefinite


def tsallis_entropy(
    mat_x: np.ndarray | cvxpy.Expression,
    t: float,
) -> float:
    r"""Compute the Tsallis entropy \(S_t(A)\) for a PSD matrix \(A\) [@fawzi2015matrixgeometric].

    For \(t \in [0, 1]\), the Tsallis entropy is defined by

    \[
        S_t(A) = \frac{1}{t}\operatorname{tr}\!\bigl(A^{1-t} - A\bigr).
    \]

    This function uses the natural logarithm implicitly through the limit
    \(S(A) = -\operatorname{tr}(A \log A)\): as \(t \to 0^+\),

    \[
        \lim_{t \to 0^+} S_t(A) = S(A),
    \]

    and \(S_t(A) \geqslant S(A)\) for all \(t \in [0, 1]\). The map \(S_t\) is concave on
    \(\text{H}_n^{++}\) for \(t \in [0, 1]\).

    This function evaluates the formula numerically. Constant CVXPY expressions
    with a concrete ``.value`` are routed through the numeric path. Affine or
    variable CVXPY inputs are not supported; use ``tsallis_entropy_hypo_cone``
    for composition in a parent SDP. For ``t == 0``, that cone delegates to
    ``ln_quantum_entropy_hypo_cone``.

    Args:
        mat_x: A numpy array or constant CVXPY expression for a positive
            semidefinite matrix.
        t: Order parameter in the range ``[0, 1]``.

    Raises:
        ValueError: If ``mat_x`` is not a numpy array or a cvxpy expression.
        ValueError: If ``mat_x`` is not a 2D square matrix.
        ValueError: If ``t`` is not in the range ``[0, 1]``.
        ValueError: If ``mat_x`` is not positive semidefinite.
        ValueError: If a constant CVXPY expression has no numeric ``.value``.
        ValueError: If affine or variable CVXPY inputs are passed.

    Returns:
        The Tsallis entropy \(S_t(A)\) as a float.

    Examples:
        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.state_props import tsallis_entropy
        mat_x = np.diag([0.25, 0.75])
        t = 0.5
        print(tsallis_entropy(mat_x, t))
        ```

    """
    if not isinstance(mat_x, (np.ndarray, cvxpy.Expression)):
        raise ValueError("mat_x must be a numpy array or a cvxpy expression")
    _require_square_2d(mat_x, "mat_x")
    if t < 0 or t > 1:
        raise ValueError("t must be in the range [0, 1]")

    if isinstance(mat_x, np.ndarray):
        if not is_positive_semidefinite(mat_x):
            raise ValueError("mat_x must be a positive semidefinite matrix")
        if t == 0:
            return float(np.real(-np.trace(mat_x @ logm(mat_x))))
        mat_power = psd_matrix_power(mat_x, 1 - t)
        return float(np.real((np.trace(mat_power) - np.trace(mat_x)) / t))

    if isinstance(mat_x, cvxpy.Expression) and mat_x.is_constant():
        x_val = mat_x.value
        if x_val is None:
            raise ValueError(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass mat_x as a numpy.ndarray."
            )
        return tsallis_entropy(np.asarray(x_val), t)

    _reject_nonconstant_cvxpy(mat_x)
