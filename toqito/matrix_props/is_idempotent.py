"""Checks if the matrix is an idempotent matrix."""

import numpy as np

from toqito.matrix_props import is_square


def is_idempotent(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-8) -> bool:
    r"""Check if matrix is the idempotent matrix [@WikiIdemPot].

    An *idempotent matrix* is a square matrix, which, when multiplied by itself, yields itself.
    That is, the matrix \(A\) is idempotent if and only if \(A^2 = A\).

    Examples:
        The following is an example of a \(2 x 2\) idempotent matrix:

        \[
            A = \begin{pmatrix}
                3 & -6 \\
                1 & -2
            \end{pmatrix}
        \]

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import is_idempotent

        mat = np.array([[3, -6], [1, -2]])

        print(is_idempotent(mat))
        ```


        Alternatively, the following matrix

        \[
            B = \begin{pmatrix}
                    1 & 2 & 3 \\
                    4 & 5 & 6 \\
                    7 & 8 & 9
                \end{pmatrix}
        \]

        is not idempotent.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import is_idempotent

        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        print(is_idempotent(mat))
        ```

    Args:
        mat: Matrix to check.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        Return `True` if matrix is the idempotent matrix, and `False` otherwise.

    """
    if not is_square(mat):
        return False
    return np.allclose(mat, mat @ mat, rtol=rtol, atol=atol)
