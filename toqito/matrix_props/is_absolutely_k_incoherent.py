"""Checks if the matrix is absolutely $k$-incoherent."""

import cvxpy as cp
import numpy as np

from toqito.matrix_props import is_positive_semidefinite, is_square


def is_absolutely_k_incoherent(mat: np.ndarray, k: int, tol: float = 1e-15) -> bool:
    r"""Determine whether a quantum state is absolutely k-incoherent :footcite:`Johnston_2022_Absolutely`.

    Formally, for positive integers :math:`n` and :math:`k`, a mixed quantum state is said to be absolutely k-incoherent
    if :math:`U \rho U^* \in \mathbb{I}_{k, n}` for all unitary matrices :math:`U \in \text{U}(\mathbb{C}^n)`.

    This function checks if the provided density matrix is absolutely k-incoherent based on the criteria introduced in
    :footcite:`Johnston_2022_Absolutely` and the corresponding QETLAB functionality :footcite:`QETLAB_link`. When
    necessary, an SDP is set up via ``cvxpy``.

    The notion of absolute k-incoherence is connected to the notion of quantum state antidistinguishability as discussed
    in :footcite:`Johnston_2025_Tight`.

    Examples
    =========
    .. jupyter-execute::

        import numpy as np
        from toqito.matrix_props import is_absolutely_k_incoherent
        mat = np.array([[2, 1, 2],
                    [1, 2, -1],
                    [2, -1, 5]])
        is_absolutely_k_incoherent(mat, 4)

    See Also
    ========
    :py:func:`~toqito.state_props.is_antidistinguishable.is_antidistinguishable`
    :py:func:`~toqito.matrix_props.is_k_incoherent.is_k_incoherent`

    References
    ==========
    .. footbibliography::



    :param mat: Matrix to check for absolute k-incoherence.
    :param k: The positive integer indicating the absolute coherence level.
    :param tol: Tolerance for numerical comparisons (default is 1e-15).
    :raises ValueError: If the input matrix is not square.
    :return: True if the quantum state is absolutely k-incoherent, False otherwise.

    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if not is_square(mat):
        raise ValueError("Input matrix must be square.")

    n = mat.shape[0]

    # Trivial: every matrix is absolutely k-incoherent for k >= n.
    if k >= n:
        return True

    # Check that the input matrix is a valid density matrix.
    if not (is_positive_semidefinite(mat) and np.isclose(np.trace(mat), 1, atol=tol)):
        return False

    # Compute eigenvalues and rank.
    eigvals = np.linalg.eigvalsh(mat)
    rankX = np.linalg.matrix_rank(mat, tol=tol)
    lmax = np.max(eigvals)

    # Trivial: only the maximally mixed state is absolutely 1-incoherent.
    if k == 1:
        if np.all(np.abs(eigvals - (1 / n)) <= tol):
            return True
        else:
            return False

    # [1] Theorem 4: Check rank conditions.
    if rankX <= n - k:
        return False
    elif rankX == n - k + 1:
        # Check if all nonzero eigenvalues are approximately equal.
        nonzero = eigvals[np.abs(eigvals) > tol]
        if len(nonzero) > 0 and np.all(np.abs(nonzero - nonzero[0]) <= tol):
            return True

    # [1] Theorem 5: Check if the largest eigenvalue meets the condition.
    if lmax <= 1 / (n - k + 1):
        return True

    if k == 2:
        # [1] Theorem 7: Use the Frobenius norm condition.
        frob_norm_sq = np.linalg.norm(mat, "fro") ** 2
        if frob_norm_sq <= 1 / (n - 1):
            return True
        elif n <= 3:
            return False
    elif k == n - 1:
        # [1] Corollary 1: Check maximum eigenvalue condition.
        if lmax > 1 - 1 / n:
            return False
        else:
            # [1] Theorem 8: Solve an SDP to decide absolute (n-1)-incoherence.
            lam = np.sort(np.real(eigvals))[::-1]
            n_eig = len(lam)
            L = cp.Variable((n_eig, n_eig), symmetric=True)
            constraints = []
            # Constraint: L[0, 0] == -lam[0] - sum(L[0, 1:]) - sum(L[1:, 0])
            constraints.append(L[0, 0] == -lam[0] - cp.sum(L[0, 1:]) - cp.sum(L[1:, 0]))
            # For indices j = 1 to n_eig-1, enforce L[j, j] == lam[j]
            for j in range(1, n_eig):
                constraints.append(L[j, j] == lam[j])
            # L must be positive semidefinite.
            constraints.append(L >> 0)
            # Dummy objective function.
            objective = cp.Minimize(1)
            prob = cp.Problem(objective, constraints)
            opt_val = prob.solve(solver=cp.SCS, verbose=False)
            if np.isclose(opt_val, 1.0):
                return True
    return False
