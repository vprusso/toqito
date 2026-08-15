"""Checks if the matrix is a positive semidefinite matrix."""

import numpy as np

from toqito.matrix_props import is_hermitian


def is_positive_semidefinite(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check if matrix is positive semidefinite (PSD) [@wikipediadefinite].

    Args:
        mat: Matrix to check.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        Return `True` if matrix is PSD, and `False` otherwise.

    Examples:
        Consider the following matrix

        \[
            A = \begin{pmatrix}
                    1 & -1 \\
                    -1 & 1
                \end{pmatrix}
        \]

        our function indicates that this is indeed a positive semidefinite matrix.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_positive_semidefinite

        A = np.array([[1, -1], [-1, 1]])

        print(is_positive_semidefinite(A))
        ```

        Alternatively, the following example matrix \(B\) defined as

        \[
            B = \begin{pmatrix}
                    -1 & -1 \\
                    -1 & -1
                \end{pmatrix}
        \]

        is not positive semidefinite.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_positive_semidefinite

        B = np.array([[-1, -1], [-1, -1]])

        print(is_positive_semidefinite(B))
        ```

    """
    if not is_hermitian(mat, rtol, atol):
        return False
    if mat.shape[0] == 0:
        return True

    # Fast path. A Cholesky factorization succeeds exactly when the matrix is
    # positive definite, and costs roughly a third to a fifth of an
    # eigendecomposition. Density matrices are frequently rank deficient (a pure
    # state has rank one), so factor `mat + atol * I` rather than `mat`: success
    # then means every eigenvalue exceeds `-atol`, which implies the tolerance
    # test below, since `rtol * scale` is non-negative. A failure proves nothing,
    # because Cholesky can also fail on a positive semidefinite matrix that is
    # merely ill conditioned, so fall through to the exact test in that case.
    try:
        np.linalg.cholesky(mat + atol * np.eye(mat.shape[0], dtype=mat.dtype))
        return True
    except np.linalg.LinAlgError:
        pass

    evals = np.linalg.eigvalsh(mat)
    if evals.size == 0:
        return True
    scale = max(1.0, np.max(np.abs(evals)))
    return bool(np.min(evals) >= -(atol + rtol * scale))
