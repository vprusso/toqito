"""Checks if the matrix is a positive definite matrix."""

import numpy as np

from toqito.matrix_props import is_hermitian


def is_positive_definite(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check if matrix is positive definite (PD) [@wikipediadefinite].

    Args:
        mat: Matrix to check.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        Return `True` if matrix is positive definite, and `False` otherwise.

    Examples:
        Consider the following matrix

        \[
            A = \begin{pmatrix}
                    2 & -1 & 0 \\
                    -1 & 2 & -1 \\
                    0 & -1 & 2
                \end{pmatrix}
        \]

        our function indicates that this is indeed a positive definite matrix.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_positive_definite

        A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])

        print(is_positive_definite(A))
        ```

        Alternatively, the following example matrix \(B\) defined as

        \[
            B = \begin{pmatrix}
                    -1 & -1 \\
                    -1 & -1
                \end{pmatrix}
        \]

        is not positive definite.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_positive_definite

        B = np.array([[-1, -1], [-1, -1]])

        print(is_positive_definite(B))
        ```

        !!! See Also
            [`is_positive_semidefinite`][toqito.matrix_props.is_positive_semidefinite.is_positive_semidefinite]

    """
    if is_hermitian(mat, rtol, atol):
        try:
            # Cholesky decomp requires that the matrix in question is
            # positive-definite. It will throw an error if this is not the case
            # that we catch here.
            np.linalg.cholesky(mat)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False
