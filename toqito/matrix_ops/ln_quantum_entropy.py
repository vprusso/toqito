r"""Quantum (von Neumann) entropy with natural logarithm: \(-\operatorname{tr}(X \log X)\) for PSD \(X\)."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np
from scipy.linalg import logm

from toqito.matrix_ops._cone_utils import _require_square_2d
from toqito.matrix_ops.operator_relative_entropy_epi_cone import (
    operator_relative_entropy_epi_cone,
)
from toqito.matrix_props import is_positive_semidefinite


def ln_quantum_entropy(
    mat_x: np.ndarray | cvxpy.Expression, m: int = 3, k: int = 3, apx: int = 0
) -> float:
    r"""Compute the quantum entropy \(-\operatorname{tr}(X \log X)\) for PSD \(X\).

    Note that this function uses the natural logarithm (base e) and not the base-2 logarithm.

    Args:
        mat_x: The PSD matrix to compute the quantum entropy of.
        m: The number of quadrature nodes to use.
        k: The number of square-roots to take.
        apx: The approximation to use.

    Raises:
        ValueError: If mat_x is not a numpy array or a cvxpy expression.
        ValueError: If mat_x is not a 2D square matrix.
        ValueError: If m is not at least 1.
        ValueError: If k is not at least 1.
        ValueError: If apx is not -1, 0, or 1.
        ValueError: If mat_x is not a positive semidefinite matrix.
        ValueError: If mat_x is a non-affine cvxpy expression.
        ValueError: If mat_x has no numeric initial value.


    Returns:
        The quantum entropy of the matrix as a float.

    """
    if not isinstance(mat_x, (np.ndarray, cvxpy.Expression)):
        raise ValueError("mat_x must be a numpy array or a cvxpy expression")
    _require_square_2d(mat_x, "mat_x")
    if m < 1:
        raise ValueError("m must be at least 1")
    if k < 1:
        raise ValueError("k must be at least 1")
    if apx not in (-1, 0, 1):
        raise ValueError("apx must be -1, 0, or 1")

    if isinstance(mat_x, np.ndarray):
        if not is_positive_semidefinite(mat_x):
            raise ValueError("mat_x must be a positive semidefinite matrix")
        return float(np.real(-np.trace(mat_x @ logm(mat_x))))
    if isinstance(mat_x, cvxpy.Expression) and mat_x.is_constant():
        x_val = mat_x.value
        if x_val is None:
            raise ValueError(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass mat_x as a numpy.ndarray."
            )
        return ln_quantum_entropy(np.asarray(x_val), m, k, apx)

    if not mat_x.is_affine():
        raise ValueError("mat_x must be an affine CVXPY expression.")
    if mat_x.value is None:
        raise ValueError(
            "Affine mat_x has no numeric initial value; set `.value` for PSD checks."
        )
    if not is_positive_semidefinite(mat_x.value):
        raise ValueError("mat_x must be positive semidefinite at the initial value.")

    n = int(mat_x.shape[0])
    is_cplx = np.any(np.imag(mat_x.value) != 0)
    if is_cplx:
        tau = cvxpy.Variable((n, n), hermitian=True)
    else:
        tau = cvxpy.Variable((n, n), symmetric=True)

    eye_n = cvxpy.Constant(np.eye(n))
    # H(X) = -tr[D_op(X||I)]; flip apx so cone bounds match entropy bounds.
    cons = operator_relative_entropy_epi_cone(
        mat_x,
        eye_n,
        tau,
        m=m,
        k=k,
        e=np.eye(n),
        apx=-apx,
        hermitian=is_cplx,
    )
    obj = -cvxpy.trace(tau)
    if is_cplx:
        obj = cvxpy.real(obj)
    prob = cvxpy.Problem(cvxpy.Maximize(obj), cons)
    return prob.solve(solver=cvxpy.SCS, verbose=False)
