"""Checks if the matrix is circulant."""

import numpy as np


def is_circulant(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Determine if matrix is circulant [@wikipediacirculant].

    A circulant matrix is a square matrix in which all row vectors are composed
    of the same elements and each row vector is rotated one element to the right
    relative to the preceding row vector.

    Args:
        mat: Matrix to check the circulancy of.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        Return `True` if `mat` is circulant; `False` otherwise.

    Examples:
        Consider the following matrix:

        \[
            C = \begin{pmatrix}
                    4 & 1 & 2 & 3 \\
                    3 & 4 & 1 & 2 \\
                    2 & 3 & 4 & 1 \\
                    1 & 2 & 3 & 4
                \end{pmatrix}
        \]

        As can be seen, this matrix is circulant. We can verify this in
        `|toqito⟩` as

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_circulant

        mat = np.array([[4, 1, 2, 3], [3, 4, 1, 2], [2, 3, 4, 1], [1, 2, 3, 4]])

        print(is_circulant(mat))
        ```

    """
    n, m = mat.shape
    if n != m:
        return False

    for i in range(n - 1):
        row = mat[i + 1]
        shifted = np.roll(mat[i], 1)
        if not np.allclose(row, shifted, rtol=rtol, atol=atol):
            return False
    return True
