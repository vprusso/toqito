"""Checks if the matrix is absolutely $k$-incoherent."""

import numpy as np

from toqito.matrix_props import is_positive_semidefinite, is_square


def is_absolutely_k_incoherent(mat: np.ndarray, k: int, tol: float = 1e-15) -> bool:
    r"""Determine whether a quantum state is absolutely k-incoherent [@johnston2022absolutely].

    Formally, for positive integers \(n\) and \(k\), a mixed quantum state is said to be absolutely k-incoherent
    if \(U \rho U^* \in \mathbb{I}_{k, n}\) for all unitary matrices \(U \in \text{U}(\mathbb{C}^n)\).

    This function checks if the provided density matrix is absolutely k-incoherent based on the criteria introduced in
    [@johnston2022absolutely] and the corresponding QETLAB functionality [@qetlablink]. For
    :code:`k = n - 1` the SDP characterization of [@johnston2022absolutely] (Theorem 8) reduces to a
    closed-form spectral condition, which is used directly.

    The notion of absolute k-incoherence is connected to the notion of quantum state antidistinguishability as discussed
    in [@johnston2025tight].

    Args:
        mat: Matrix to check for absolute k-incoherence.
        k: The positive integer indicating the absolute coherence level.
        tol: Tolerance for numerical comparisons (default is 1e-15).

    Returns:
        True if the quantum state is absolutely k-incoherent, False otherwise.

    Raises:
        ValueError: If the input matrix is not square.

    Examples:
        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_absolutely_k_incoherent

        mat = np.array([[2, 1, 2],
                    [1, 2, -1],
                    [2, -1, 5]])
        print(is_absolutely_k_incoherent(mat, 4))
        ```

        !!! See
            [is_antidistinguishable()][toqito.state_props.is_antidistinguishable.is_antidistinguishable],
            [is_k_incoherent()][toqito.matrix_props.is_k_incoherent.is_k_incoherent]

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
    # For a Hermitian matrix the singular values equal the absolute eigenvalues, so the rank (number
    # of singular values above ``tol``) can be read off the already-computed eigenvalues.
    rankX = int(np.count_nonzero(np.abs(eigvals) > tol))
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
        # [1] Theorem 8 gives an SDP characterization of absolute (n-1)-incoherence. That SDP
        # admits a closed-form solution: factoring the Gram-type matrix in the SDP shows that it
        # is feasible if and only if
        #     sqrt(lmax) <= sum of sqrt of the remaining eigenvalues.
        # This spectral test is exact, so no SDP solve is needed (solving the feasibility SDP
        # numerically is also unreliable; first-order solvers can report false feasibility).
        lam = np.sort(np.real(eigvals))[::-1]
        return bool(np.sqrt(lam[0]) <= np.sum(np.sqrt(np.clip(lam[1:], 0, None))) + np.sqrt(tol))
    return False
