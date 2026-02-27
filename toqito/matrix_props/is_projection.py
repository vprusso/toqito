"""Checks if the matrix is a projection matrix."""

import numpy as np

from toqito.matrix_props import is_square


def is_projection(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check if matrix is a projection matrix [@WikiProjMat].

    A matrix is a projection matrix if it is positive semidefinite (PSD) and if

    \[
        \begin{equation}
            X^2 = X
        \end{equation}
    \]

    where \(X\) is the matrix in question.

    Examples:
        Consider the following matrix

        \[
            A = \begin{pmatrix}
                    0 & 1 \\
                    0 & 1
                \end{pmatrix}
        \]

        our function indicates that this is indeed a projection matrix.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import is_projection

        A = np.array([[0, 1], [0, 1]])

        print(is_projection(A))
        ```

        Alternatively, the following example matrix \(B\) defined as

        \[
            B = \begin{pmatrix}
                    -1 & -1 \\
                    -1 & -1
                \end{pmatrix}
        \]

        is not positive definite.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import is_projection

        B = np.array([[-1, -1], [-1, -1]])

        print(is_projection(B))
        ```

    Args:
        mat: Matrix to check.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        Return `True` if matrix is a projection matrix, and `False` otherwise.

    """
    if not is_square(mat):
        return False
    return np.allclose(np.linalg.matrix_power(mat, 2), mat, rtol=rtol, atol=atol)
