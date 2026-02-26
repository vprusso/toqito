"""Checks if the matrix is an identity matrix."""

import numpy as np

from toqito.matrix_props import is_square


def is_identity(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-8) -> bool:
    r"""Check if matrix is the identity matrix [@WikiIden].

    For dimension \(n\), the \(n \times n\) identity matrix is defined as

    \[
        I_n =
        \begin{pmatrix}
            1 & 0 & 0 & \ldots & 0 \\
            0 & 1 & 0 & \ldots & 0 \\
            0 & 0 & 1 & \ldots & 0 \\
            \vdots & \vdots & \vdots & \ddots & \vdots \\
            0 & 0 & 0 & \ldots & 1
        \end{pmatrix}.
    \]

    Examples:
    Consider the following matrix:

    \[
        A = \begin{pmatrix}
                1 & 0 & 0 \\
                0 & 1 & 0 \\
                0 & 0 & 1
            \end{pmatrix}
    \]

    our function indicates that this is indeed the identity matrix of dimension
    3.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_props import is_identity

    mat = np.eye(3)

    print(is_identity(mat))
    ```

    Alternatively, the following example matrix \(B\) defined as

    \[
        B = \begin{pmatrix}
                1 & 2 & 3 \\
                4 & 5 & 6 \\
                7 & 8 & 9
            \end{pmatrix}
    \]

    is not an identity matrix.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_props import is_identity

    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    print(is_identity(mat))
    ```

    Args:
        mat: Matrix to check.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        Return `True` if matrix is the identity matrix, and `False` otherwise.

    """
    if not is_square(mat):
        return False
    id_mat = np.eye(len(mat))
    return np.allclose(mat, id_mat, rtol=rtol, atol=atol)
