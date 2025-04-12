"""Checks if the matrix is $k$-incoherent."""

import numpy as np
import cvxpy as cp
from itertools import combinations

from toqito.matrix_props import is_positive_semidefinite, is_square
from toqito.matrices.comparison import comparison


def is_k_incoherent(mat: np.ndarray, k: int, tol: float = 1e-15) -> bool:
    r"""Determine whether a quantum state is k-incoherent :cite:`Johnston_2022_Absolutely`.

    This function checks if the provided density matrix :code:`mat` is k-incoherent. It returns True if :code:`mat` is
    k-incoherent and False if :code:`mat` is not.

    The function first handles trivial cases. Then it computes the comparison matrix (via
    :func:`matrices.comparison.comparison`) and performs further tests based on the trace of :code:`mat @ mat` and a
    dephasing channel. If no decision is reached, the function recurses by checking incoherence for k-1.  Finally, if
    still indeterminate, an SDP is formulated to decide incoherence.

    :param mat: Density matrix to test.
    :param k: The positive integer coherence level.
    :param tol: Tolerance for numerical comparisons (default is 1e-15).
    :raises ValueError: If k ≤ 0 or if :code:`mat` is not square.
    :return: True if :code:`mat` is k-incoherent, False otherwise.

    See Also
    ========
    :func:`toqito.state_props.is_antidistinguishable`
    :func:`toqito.matrix_props.is_absolutely_k_incoherent`

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if not is_square(mat):
        raise ValueError("Input matrix must be square.")
    d = mat.shape[0]

    # Trivial: every state is d-incoherent.
    if k >= d:
        return True

    # If :code:`mat` is diagonal, it is declared k-incoherent.
    if np.allclose(mat, np.diag(np.diag(mat)), atol=tol):
        return True

    # For k == 1, only diagonal states are 1-incoherent.
    if k == 1:
        return False

    # [1] Theorem 1: Use the comparison matrix.
    M = comparison(mat)
    if is_positive_semidefinite(M, atol=tol):
        return True
    elif k == 2:
        return False

    # :cite:`Johnston_2022_Absolutely` (8): Check if trace(mat^2) <= 1/(d - 1) (for k > 2).
    if k > 2 and np.trace(mat @ mat) <= 1 / (d - 1):
        return True

    # :cite:`Johnston_2022_Absolutely` (7): Apply dephasing channel.
    test = ((d - k) / (d - 1)) * np.diag(np.diag(mat))
    if is_positive_semidefinite(mat - test, atol=tol):
        return True

    # Hierarchical recursion: for k >= 2 check incoherence for k-1.
    rec = is_k_incoherent(mat, k - 1, atol=tol)
    if rec is not None and rec is not False:
        return rec

    # Fallback: use an SDP to decide incoherence.
    # We follow the MATLAB method via projections onto k-element subsets.
    n = d  # for clarity, n == d
    idx_sets = list(combinations(range(n), k))
    s = len(idx_sets)
    A_vars = [cp.Variable((k, k), hermitian=True) for _ in range(s)]
    constraints = []
    P_expr = 0
    for idx, A_j in zip(idx_sets, A_vars):
        # Build the projection matrix (constant, shape (k, n)).
        proj = np.zeros((k, n))
        for i, j in enumerate(idx):
            proj[i, j] = 1.0
        constraints.append(A_j >> 0)
        P_expr = P_expr + proj.T @ A_j @ proj
    constraints.append(mat == P_expr)
    prob = cp.Problem(cp.Minimize(1), constraints)
    try:
        opt_val = prob.solve(solver=cp.SCS, verbose=False)
    except cp.SolverError:
        return False
    if opt_val is None:
        return False
    # MATLAB sets ikinc = 1 - min(cvx_optval, 1); here we interpret an optimum near 1 as True.
    return np.isclose(1 - min(opt_val, 1), 1.0)
