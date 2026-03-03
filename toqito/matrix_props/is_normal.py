"""Checks if the matrix is a normal matrix."""

import numpy as np

from toqito.matrix_props import is_square


def is_normal(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Determine if a matrix is normal [@wikipedianormal].

    A matrix is normal if it commutes with its adjoint

    \[
        \begin{equation}
            [X, X^*] = 0,
        \end{equation}
    \]

    or, equivalently if

    \[
        \begin{equation}
            X^* X = X X^*
        \end{equation}.
    \]

    Examples:
        Consider the following matrix

        \[
            A = \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 1
                \end{pmatrix}
        \]

        our function indicates that this is indeed a normal matrix.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import is_normal

        A = np.identity(4)

        print(is_normal(A))
        ```

        Alternatively, the following example matrix \(B\) defined as

        \[
            B = \begin{pmatrix}
                    1 & 2 & 3 \\
                    4 & 5 & 6 \\
                    7 & 8 & 9
                \end{pmatrix}
        \]

        is not normal.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import is_normal

        B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        print(is_normal(B))
        ```

    Args:
        mat: The matrix to check.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        Returns `True` if the matrix is normal and `False` otherwise.

    """
    if not is_square(mat):
        return False
    return np.allclose(mat @ mat.conj().T, mat.conj().T @ mat, rtol=rtol, atol=atol)
