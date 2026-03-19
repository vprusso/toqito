"""Compute the nonnegative rank of a nonnegative matrix."""

import numpy as np
from scipy.optimize import nnls

from toqito.matrix_props import is_nonnegative


def nonnegative_rank(mat: np.ndarray, max_rank: int | None = None, tol: float = 1e-6) -> int | None:
    r"""Compute the nonnegative rank of a nonnegative matrix.

    The nonnegative rank of a matrix \(A \in \mathbb{R}_{\geq 0}^{n \times m}\) is the
    smallest integer \(k\) such that there exist nonnegative matrices
    \(L \in \mathbb{R}_{\geq 0}^{n \times k}\) and
    \(R \in \mathbb{R}_{\geq 0}^{k \times m}\) satisfying \(A = LR\).

    The nonnegative rank satisfies

    \[
        \text{rank}(A) \leq \text{rank}_+(A) \leq \min(n, m).
    \]

    This function checks feasibility for increasing values of \(k\) starting from
    \(\text{rank}(A)\) using alternating nonnegative least squares (ANLS).

    For more information, see [@barioli2003maximal].

    Args:
        mat: A nonnegative matrix.
        max_rank: Maximum rank to check. Defaults to `min(n, m)`.
        tol: Numerical tolerance for the factorization residual. Default 1e-6.

    Returns:
        The nonnegative rank if found within `max_rank`, or `None` otherwise.

    Raises:
        ValueError: If the matrix contains negative entries.

    Examples:
        The nonnegative rank of the identity matrix equals its dimension.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import nonnegative_rank

        print(nonnegative_rank(np.eye(3)))
        ```

        A rank-1 nonnegative matrix has nonnegative rank 1.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import nonnegative_rank

        A = np.array([[1, 2], [2, 4]])
        print(nonnegative_rank(A))
        ```

    """
    if not is_nonnegative(mat):
        raise ValueError("Matrix must be nonnegative.")

    mat = np.asarray(mat, dtype=float)
    n, m = mat.shape

    if np.allclose(mat, 0, atol=tol):
        return 0

    standard_rank = np.linalg.matrix_rank(mat, tol=tol)

    if max_rank is None:
        max_rank = min(n, m)

    for k in range(standard_rank, max_rank + 1):
        if _check_nn_rank_anls(mat, k, tol):
            return k
    return None


def _check_nn_rank_anls(mat: np.ndarray, k: int, tol: float, max_iter: int = 200, n_restarts: int = 5) -> bool:
    """Check if a nonneg factorization of rank k exists via alternating nonneg least squares.

    Runs multiple random restarts to avoid local minima.

    Args:
        mat: Nonneg matrix of shape (n, m).
        k: Target rank.
        tol: Tolerance for residual.
        max_iter: Maximum ANLS iterations per restart.
        n_restarts: Number of random restarts.

    Returns:
        True if a rank-k nonneg factorization exists within tolerance.

    """
    n, m = mat.shape
    best_residual = float("inf")

    for seed in range(n_restarts):
        rng = np.random.default_rng(seed)
        # Initialize R randomly.
        R = rng.random((k, m)) + 1e-10

        for _ in range(max_iter):
            # Fix R, solve for L: min ||A - L @ R||_F^2 s.t. L >= 0.
            # Solve column-by-column: for each row of A, solve A[i,:] = L[i,:] @ R.
            L = np.zeros((n, k))
            for i in range(n):
                L[i, :], _ = nnls(R.T, mat[i, :])

            # Fix L, solve for R: min ||A - L @ R||_F^2 s.t. R >= 0.
            R = np.zeros((k, m))
            for j in range(m):
                R[:, j], _ = nnls(L, mat[:, j])

            residual = np.linalg.norm(mat - L @ R, "fro") / max(np.linalg.norm(mat, "fro"), 1e-15)
            if residual < tol:
                return True

        best_residual = min(best_residual, residual)

    return best_residual < tol
