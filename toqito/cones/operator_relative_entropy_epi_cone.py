"""CVXPY constraints for the operator relative entropy epigraph cone."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import cvxpy
import numpy as np

from toqito.matrix_ops._cone_utils import _require_square_2d, _symmetric_like_variable
from toqito.matrix_ops.geometric_mean_hypo_cone import geometric_mean_hypo_cone


def _gauss_legendre(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute Gauss-Legendre quadrature nodes and weights on [0,1].

    Based on the implementation in [@trefethen2008gauss].
    """
    if m < 1:
        raise ValueError("m must be at least 1.")
    if m == 1:
        return np.array([0.5]), np.array([1.0])
    k = np.arange(1, m, dtype=np.float64)
    beta = 0.5 / np.sqrt(1 - (2 * k) ** (-2))
    T = np.diag(beta, 1) + np.diag(beta, -1)
    nodes_m11, V = np.linalg.eigh(T)
    weights_m11 = 2 * V[0, :] ** 2
    nodes = (nodes_m11 + 1) / 2
    weights = weights_m11 / 2
    return nodes, weights


def _gauss_radau(m: int, endpoint: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute Gauss-Radau quadrature nodes and weights on [0,1].

    endpoint should be either 0 or 1.
    """
    if m < 1:
        raise ValueError("m must be at least 1.")
    if endpoint not in [0, 1]:
        raise ValueError("endpoint must be either 0 or 1.")
    if m == 1:
        return np.array([endpoint]), np.array([1.0])
    k = np.arange(1, m, dtype=np.float64)
    beta = 0.5 / np.sqrt(1 - (2 * k) ** (-2))
    a = np.sign(endpoint - 0.5) * (1 - beta[-1] ** 2 * (2 * m - 3) / (m - 1))
    T = np.diag(beta, 1) + np.diag(beta, -1)
    T[m - 1, m - 1] = a
    nodes_m11, V = np.linalg.eigh(T)
    weights_m11 = 2 * V[0, :] ** 2
    nodes = (nodes_m11 + 1) / 2
    weights = weights_m11 / 2
    return nodes, weights


def operator_relative_entropy_epi_cone(
    X: cvxpy.Expression,
    Y: cvxpy.Expression,
    TAU: cvxpy.Expression,
    *,
    m: int = 3,
    k: int = 3,
    e: np.ndarray | None = None,
    apx: int = 0,
    hermitian: bool = False,
) -> list[cvxpy.Constraint]:
    r"""Return CVXPY constraints for the operator relative entropy epigraph cone [@fawzi2017matrixlogarithm].

    The set of matrices that satisfy the constraints are `X`, `Y`, `TAU` triples such that

    \[
        X^{1/2} \log\!\bigl(X^{1/2} Y^{-1} X^{1/2}\bigr) X^{1/2} \preceq \mathrm{TAU}
    \]

    in the PSD order, where ``log`` is the matrix logarithm.
    Auxiliary variables ``Z`` and quadrature slices `T_i` are introduced internally.

    Args:
        X: A cvxpy expression representing a matrix.
        Y: A cvxpy expression representing a matrix.
        TAU: A cvxpy expression representing a matrix.
        m: The number of quadrature nodes to use.
        k: The number of square-roots to take.
        e: A numpy array representing a matrix.
        apx: Log via quadrature: ``-1`` uses left-endpoint Gauss-Radau (lower bound on
            the matrix log), ``0`` uses Gauss-Legendre on ``[0, 1]`` (two-sided midpoint
            rule), ``1`` uses right-endpoint Gauss--Radau (upper bound on the matrix log).
        hermitian: Whether the matrices are Hermitian or symmetric.

    Raises:
        ValueError: If the matrices are not square.
        ValueError: If the matrices are not the same size.
        ValueError: If e is not 2D.
        ValueError: If the number of rows of e does not match the size of X.
        ValueError: If the size of TAU does not match the size of e.
        ValueError: If m or k is less than 1.
        ValueError: If apx is not -1, 0, or 1.
        ValueError: If TAU is not r x r with r = e.shape[1].

    Returns:
        A list of CVX constraints.

    """
    _require_square_2d(X, "X")
    _require_square_2d(Y, "Y")
    _require_square_2d(TAU, "TAU")
    n = int(X.shape[0])
    if int(Y.shape[0]) != n or int(Y.shape[1]) != n:
        raise ValueError("Y must have the same shape as X.")
    if e is None:
        e = np.eye(n)
    e = np.asarray(e)
    if e.ndim != 2:
        raise ValueError("e must be 2D.")
    if int(e.shape[0]) != n:
        raise ValueError("The number of rows of e must match X.")
    r = int(e.shape[1])
    if int(TAU.shape[0]) != r or int(TAU.shape[1]) != r:
        raise ValueError("TAU must be r x r with r = e.shape[1].")

    if m < 1 or k < 1:
        raise ValueError("m and k must be at least 1.")
    if apx not in [-1, 0, 1]:
        raise ValueError("apx must be either -1, 0, or 1.")
    if apx == 0:
        s, w = _gauss_legendre(m)
    elif apx == 1:
        s, w = _gauss_radau(m, 1)
    else:
        s, w = _gauss_radau(m, 0)

    w = np.ravel(np.asarray(w, dtype=np.float64))
    s = np.ravel(np.asarray(s, dtype=np.float64))

    e_h = e.conj().T

    z_var = _symmetric_like_variable(n, hermitian=hermitian)
    t_pages = [_symmetric_like_variable(r, hermitian=hermitian) for _ in range(m)]

    constraints: list[cvxpy.Constraint] = []
    constraints.extend(
        geometric_mean_hypo_cone(
            X,
            Y,
            z_var,
            float(2.0 ** (-k)),
            fullhyp=False,
            hermitian=hermitian,
        )
    )

    for ii in range(m):
        si = float(s[ii])
        wi = float(w[ii])
        ti = t_pages[ii]
        if si < 1e-6:
            quad = e_h @ (z_var - X) @ e
            constraints.append(quad - ti / wi >> 0)
        else:
            top_left = e_h @ X @ e - (si / wi) * ti
            top_right = e_h @ X
            bot_left = X @ e
            bot_right = (1.0 - si) * X + si * z_var
            constraints.append(
                cvxpy.bmat([[top_left, top_right], [bot_left, bot_right]]) >> 0
            )

    t_sum = sum(t_pages)
    constraints.append((2.0**k) * t_sum + TAU >> 0)

    return constraints
