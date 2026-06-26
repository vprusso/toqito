"""Checks if the matrix is a diagonal matrix."""

import numpy as np

from toqito.matrix_props import is_square


def is_diagonal(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Determine if a matrix is diagonal [@wikipediadiagonal].

    A matrix is diagonal if the matrix is square and the off-diagonal elements are all (approximately) zero.

    Args:
        mat: The matrix to check.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        Returns `True` if the matrix is diagonal and `False` otherwise.

    Examples:
        The following is an example of a 3-by-3 diagonal matrix:

        \[
            \begin{equation}
                \begin{pmatrix}
                    1 & 0 & 0 \\
                    0 & 2 & 0 \\
                    0 & 0 & 3
                \end{pmatrix}
            \end{equation}
        \]

        Consider the following diagonal matrix:

        \[
            A = \begin{pmatrix}
                    1 & 0 \\
                    0 & 1
                \end{pmatrix}.
        \]

        Our function indicates that this is indeed a diagonal matrix:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_diagonal

        A = np.array([[1, 0], [0, 1]])

        print(is_diagonal(A))
        ```

        Alternatively, the following example matrix

        \[
            B = \begin{pmatrix}
                    1 & 2 \\
                    3 & 4
                \end{pmatrix}
        \]

        is not diagonal, as shown using `|toqito⟩`.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_diagonal

        B = np.array([[1, 2], [3, 4]])

        print(is_diagonal(B))
        ```

    """
    if not is_square(mat):
        return False
    # The matrix is diagonal when it equals the matrix of just its diagonal, up to tolerance. This handles
    # near-zero off-diagonal noise and the 1x1 case, unlike an exact comparison.
    return bool(np.allclose(mat, np.diag(np.diag(mat)), rtol=rtol, atol=atol))
