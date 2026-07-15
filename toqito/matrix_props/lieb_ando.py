r"""Computes \(f(A, B, K, t) = \operatorname{tr}(K^{\dagger} A^{1-t} K B^{t})\) for PSD \(A\) and \(B\)."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np

from toqito.cones._utils import (
    _reject_nonconstant_cvxpy,
    _require_2d,
    _require_square_2d,
)
from toqito.matrix_ops import psd_matrix_power
from toqito.matrix_props import is_positive_semidefinite


def lieb_ando(
    mat_a: np.ndarray | cvxpy.Expression,
    mat_b: np.ndarray | cvxpy.Expression,
    mat_k: np.ndarray,
    t: float,
) -> float:
    r"""Compute \(f(A, B, K, t) = \operatorname{tr}\!\bigl(K^{\dagger} A^{1-t} K B^{t}\bigr)\) for PSD matrices.

    Here \(A=\) ``mat_a``, \(B=\) ``mat_b``, and \(K=\) ``mat_k``. For real data,
    \(K^{\dagger} = K^{\top}\). The map is concave in \((A, B)\) for
    \(t \in [0, 1]\) and convex for \(t \in [-1, 0] \cup [1, 2]\)
    [@fawzi2015matrixgeometric].

    This function evaluates the formula numerically. Constant CVXPY expressions
    with a concrete ``.value`` are routed through the numeric path. Affine or
    variable CVXPY inputs are not supported; use ``lieb_ando_hypo_cone`` (for
    ``t`` in ``[0, 1]``) or ``lieb_ando_epi_cone`` (for ``t`` in ``[-1, 0]`` or
    ``[1, 2]``) for composition in a parent SDP.

    Args:
        mat_a: A numpy array or constant CVXPY expression for the first PSD matrix.
        mat_b: A numpy array or constant CVXPY expression for the second PSD matrix.
        mat_k: The fixed numpy weight matrix.
        t: The Lieb-Ando exponent.

    Returns:
        The value of the function as a float.

    Raises:
        TypeError: If ``mat_a`` or ``mat_b`` is not a numpy.ndarray or a cvxpy expression.
        TypeError: If ``mat_k`` is not a numpy.ndarray.
        ValueError: If ``mat_a`` is not 2D or not square.
        ValueError: If ``mat_b`` is not 2D or not square.
        ValueError: If ``mat_k`` is not 2D.
        ValueError: If ``mat_k`` has incompatible shape with ``mat_a`` and ``mat_b``.
        ValueError: If ``mat_a`` and ``mat_b`` are not positive semidefinite.
        ValueError: If a constant CVXPY expression has no numeric ``.value``.
        ValueError: If affine or variable CVXPY inputs are passed.

    Examples:
        ```python
        import numpy as np
        from toqito.matrix_props import lieb_ando
        mat_a = np.array([[2.0, 1.0], [1.0, 2.0]])
        mat_b = np.array([[2.0, 1.0], [1.0, 2.0]])
        mat_k = np.eye(2)
        t = 0.5
        print(lieb_ando(mat_a, mat_b, mat_k, t))
        ```

    """
    if not isinstance(mat_a, (np.ndarray, cvxpy.Expression)):
        raise TypeError("mat_a must be a numpy.ndarray or a cvxpy expression.")
    if not isinstance(mat_b, (np.ndarray, cvxpy.Expression)):
        raise TypeError("mat_b must be a numpy.ndarray or a cvxpy expression.")
    if not isinstance(mat_k, np.ndarray):
        raise TypeError("mat_k must be a numpy.ndarray.")

    _require_square_2d(mat_a, "mat_a")
    _require_square_2d(mat_b, "mat_b")
    _require_2d(mat_k, "mat_k")
    if mat_k.shape[0] != mat_a.shape[0] or mat_k.shape[1] != mat_b.shape[1]:
        raise ValueError(
            "mat_k must have the same number of rows as mat_a and the same number of columns as mat_b."
        )

    if isinstance(mat_a, cvxpy.Expression) and mat_a.is_constant():
        a_val = mat_a.value
        if a_val is None:
            raise ValueError(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass mat_a as a numpy.ndarray."
            )
        return lieb_ando(np.asarray(a_val), mat_b, mat_k, t)

    if isinstance(mat_b, cvxpy.Expression) and mat_b.is_constant():
        b_val = mat_b.value
        if b_val is None:
            raise ValueError(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass mat_b as a numpy.ndarray."
            )
        return lieb_ando(mat_a, np.asarray(b_val), mat_k, t)

    if isinstance(mat_a, np.ndarray) and isinstance(mat_b, np.ndarray):
        if not is_positive_semidefinite(mat_a) or not is_positive_semidefinite(mat_b):
            raise ValueError("mat_a and mat_b must be positive semidefinite.")
        a_raised = psd_matrix_power(mat_a, 1 - t)
        b_raised = psd_matrix_power(mat_b, t)
        return float(np.real(np.trace(mat_k.conj().T @ a_raised @ mat_k @ b_raised)))

    _reject_nonconstant_cvxpy(mat_a, mat_b)
