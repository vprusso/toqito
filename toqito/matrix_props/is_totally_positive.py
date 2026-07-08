"""Checks if the matrix is totally positive."""

import warnings
from itertools import combinations

import numpy as np


def is_totally_positive(
    mat: np.ndarray,
    rtol: float = 0.0,
    atol: float = 1e-6,
    sub_sizes: list | None = None,
    *,
    tol: float | None = None,
) -> bool:
    r"""Determine whether a matrix is totally positive [@wikipediatotallypositive].

    A totally positive matrix is a square matrix where all the minors are positive. Equivalently, the determinant of
    every square submatrix is a positive number.

    Args:
        mat: Matrix to check.
        rtol: Relative tolerance applied (against the largest matrix entry) when testing entries
            for negativity. Defaults to ``0.0``, so by default only ``atol`` is used.
        atol: Absolute tolerance parameter (default 1e-06).
        sub_sizes: List of sizes of submatrices to consider. Default is all sizes up to `min(mat.shape)`.
        tol: Deprecated alias retained for backward compatibility; if given it sets ``atol``.

    Returns:
        Return `True` if matrix is totally positive, and `False` otherwise.

    Examples:
        Consider the matrix

        \[
            X = \begin{pmatrix}
                5 & 2 \\
                2 & 1
            \end{pmatrix}
        \]

        To determine if this matrix is totally positive, we need to check the positivity of all of its minors. The 1x1
        minors are simply the individual entries of the matrix. For \(X\), these are

        \[
            \begin{equation}
                \begin{aligned}
                    X_{1,1} &= 5 \\
                    X_{1,2} &= 2 \\
                    X_{2,1} &= 2 \\
                    X_{2,2} &= 1 \\
                \end{aligned}
            \end{equation}
        \]

        Each of these entries is positive. There is only one 2x2 minor in this case, which is the determinant of
        the entire matrix \(X\). The determinant of \(X\) is calculated as:

        \[
            \text{det}(X) = 5 \times 1 - 2 \times 2 = 5 - 4 = 1
        \]

        Our function indicates that this matrix is indeed totally positive.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_totally_positive

        A = np.array([[5, 2], [2, 1]])

        print(is_totally_positive(A))
        ```

        However, the following example matrix \(B\) defined as

        \[
            B = \begin{pmatrix}
                    1 & 2 \\
                    3 & 4
                \end{pmatrix}
        \]

        is not totally positive. The determinant of \(B\) is calculated as:

        \[
            \text{det}(B) = 1 \times 4 - 2 \times 3 = 4 - 6 = -2
        \]

        Since the determinant is negative, \(B\) is not totally positive.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_totally_positive

        B = np.array([[1, 2], [3, 4]])

        print(is_totally_positive(B))
        ```

    """
    if tol is not None:
        warnings.warn(
            "`tol` is deprecated; use `atol` (and optionally `rtol`) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        atol = tol

    if mat.size == 0:
        raise ValueError("Cannot determine total positivity of an empty matrix.")

    dims = mat.shape

    # Entries are flagged as negative when they fall below the combined relative/absolute threshold.
    neg_thresh = atol + rtol * np.abs(mat).max()

    if sub_sizes is None:
        sub_sizes = range(1, min(dims) + 1)

    for j in sub_sizes:
        # Handle 1x1 determinants separately.
        if j == 1:
            r, _ = np.where(np.minimum(np.real(mat), -np.abs(np.imag(mat))) < -neg_thresh)
            if r.size > 0:
                return False

        # Handle larger determinants.
        else:
            sub_ind_r = list(combinations(range(dims[0]), j))
            sub_ind_c = list(combinations(range(dims[1]), j)) if dims[0] != dims[1] else sub_ind_r

            for kr in sub_ind_r:
                for kc in sub_ind_c:
                    d = np.linalg.det(mat[np.ix_(kr, kc)])
                    if d < atol or abs(np.imag(d)) > atol:
                        return False
    return True
