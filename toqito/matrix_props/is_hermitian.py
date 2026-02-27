"""Checks if the matrix is a Hermitian matrix."""

import numpy as np

from toqito.matrix_props import is_square


def is_hermitian(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check if matrix is Hermitian [@WikiHerm].

    A Hermitian matrix is a complex square matrix that is equal to its own conjugate transpose.

    Examples:
        Consider the following matrix:

        \[
            A = \begin{pmatrix}
                    2 & 2 +1j & 4 \\
                    2 - 1j & 3 & 1j \\
                    4 & -1j & 1
                \end{pmatrix}
        \]

        our function indicates that this is indeed a Hermitian matrix as it holds that

        \[
            A = A^*.
        \]

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import is_hermitian

        mat = np.array([[2, 2 + 1j, 4], [2 - 1j, 3, 1j], [4, -1j, 1]])

        print(is_hermitian(mat))
        ```

        Alternatively, the following example matrix \(B\) defined as

        \[
            B = \begin{pmatrix}
                    1 & 2 & 3 \\
                    4 & 5 & 6 \\
                    7 & 8 & 9
                \end{pmatrix}
        \]

        is not Hermitian.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import is_hermitian

        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        print(is_hermitian(mat))
        ```

    Args:
        mat: Matrix to check.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        Return True if matrix is Hermitian, and False otherwise.

    """
    if not is_square(mat):
        return False
    return np.allclose(mat, mat.conj().T, rtol=rtol, atol=atol)
