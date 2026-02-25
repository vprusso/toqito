"""Checks if the matrix is diagonally dominant."""

import numpy as np

from toqito.matrix_props import is_square


def is_diagonally_dominant(mat: np.ndarray, is_strict: bool = True) -> bool:
    r"""Check if matrix is diagnal dominant (DD) [@WikiDiagDom].

    A matrix is diagonally dominant if the matrix is square
    and if for every row of the matrix, the magnitude of the diagonal entry in a row is greater
    than or equal to the sum of the magnitudes of all the other (non-diagonal) entries in that row.

    Examples:
    The following is an example of a 3-by-3 diagonal matrix:

    \[
        A = \begin{pmatrix}
                2 & -1 & 0 \\
                0 & 2 & -1 \\
                0 & -1 & 2
            \end{pmatrix}
    \]

    our function indicates that this is indeed a diagonally dominant matrix.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_props import is_diagonally_dominant

    A = np.array([[2, -1, 0], [0, 2, -1], [0, -1, 2]])

    print(is_diagonally_dominant(A))
    ```

    Alternatively, the following example matrix \(B\) defined as

    \[
        B = \begin{pmatrix}
                -1 & 2 \\
                -1 & -1
            \end{pmatrix}
    \]

    is not diagonally dominant.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_props import is_diagonally_dominant

    B = np.array([[-1, 2], [-1, -1]])

    print(is_diagonally_dominant(B))
    ```

    Args:
        mat: Matrix to check.
        is_strict: Whether the inequality is strict.

    Returns:
        Return `True` if matrix is diagnally dominant, and `False` otherwise.

    """
    if not is_square(mat):
        return False

    mat = np.abs(mat)
    diag_coeffs = np.diag(mat)
    row_sum = np.sum(mat, axis=1) - diag_coeffs
    return bool(np.all(diag_coeffs > row_sum) if is_strict else np.all(diag_coeffs >= row_sum))
