"""Checks if the matrix is a permutation matrix."""

import numpy as np


def is_permutation(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Determine if a matrix is a permutation matrix [@wikipediapermutation].

    A matrix is a permutation matrix if it is square, every entry is 0 or 1, and each row and column sums to 1.

    Args:
        mat: The matrix to check.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        Returns `True` if the matrix is a permutation matrix and `False` otherwise.

    Examples:
        Consider the following permutation matrix

        \[
            A = \begin{pmatrix}
                    1 & 0 & 0 \\
                    0 & 0 & 1 \\
                    0 & 1 & 0
                \end{pmatrix}
        \]

        which is indeed a permutation matrix.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_permutation

        A = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        print(is_permutation(A))
        ```


        Alternatively, the following example matrix \(B\) defined as

        \[
            B = \begin{pmatrix}
                    1 & 0 & 0 \\
                    1 & 0 & 0 \\
                    1 & 0 & 0
                \end{pmatrix}
        \]

        has 2 columns with all zero values and is thus not a
        permutation matrix.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_permutation

        B = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])

        print(is_permutation(B))
        ```

    """
    mat = np.asarray(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False

    # Every entry must be 0 or 1 (within tolerance).
    is_zero_or_one = np.isclose(mat, 0, rtol=rtol, atol=atol) | np.isclose(mat, 1, rtol=rtol, atol=atol)
    if not np.all(is_zero_or_one):
        return False

    # Each row and each column must sum to 1.
    rows_ok = np.allclose(mat.sum(axis=1), 1, rtol=rtol, atol=atol)
    cols_ok = np.allclose(mat.sum(axis=0), 1, rtol=rtol, atol=atol)
    return bool(rows_ok and cols_ok)
