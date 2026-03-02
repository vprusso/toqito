"""Checks if the matrix is a symmetric matrix."""

import numpy as np

from toqito.matrix_props import is_square


def is_symmetric(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Determine if a matrix is symmetric [@wikipediasymmetric].

    The following 3x3 matrix is an example of a symmetric matrix:

    \[
        \begin{pmatrix}
            1 & 7 & 3 \\
            7 & 4 & -5 \\
            3 &-5 & 6
        \end{pmatrix}
    \]

    Examples:
        Consider the following matrix

        \[
            A = \begin{pmatrix}
                    1 & 7 & 3 \\
                    7 & 4 & -5 \\
                    3 & -5 & 6
                \end{pmatrix}
        \]

        our function indicates that this is indeed a symmetric matrix.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import is_symmetric

        A = np.array([[1, 7, 3], [7, 4, -5], [3, -5, 6]])

        print(is_symmetric(A))
        ```

        Alternatively, the following example matrix \(B\) defined as

        \[
            B = \begin{pmatrix}
                    1 & 2 \\
                    4 & 5
                \end{pmatrix}
        \]

        is not symmetric.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import is_symmetric

        B = np.array([[1, 2], [3, 4]])

        print(is_symmetric(B))
        ```

    Args:
        mat: The matrix to check.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        Returns `True` if the matrix is symmetric and `False` otherwise.

    """
    if not is_square(mat):
        return False
    return np.allclose(mat, mat.T, rtol=rtol, atol=atol)
