"""Checks if the matrix is a square matrix."""

import numpy as np


def is_square(mat: np.ndarray) -> bool:
    r"""Determine if a matrix is square [@WikiSqMat].

    A matrix is square if the dimensions of the rows and columns are equivalent.

    Examples:
        Consider the following matrix

        \[
            A = \begin{pmatrix}
                    1 & 2 & 3 \\
                    4 & 5 & 6 \\
                    7 & 8 & 9
                \end{pmatrix}
        \]

        our function indicates that this is indeed a square matrix.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import is_square

        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        print(is_square(A))
        ```

        Alternatively, the following example matrix \(B\) defined as

        \[
            B = \begin{pmatrix}
                    1 & 2 & 3 \\
                    4 & 5 & 6
                \end{pmatrix}
        \]

        is not square.

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import is_square

        B = np.array([[1, 2, 3], [4, 5, 6]])

        print(is_square(B))
        ```

    Raises:
        ValueError: If variable is not a matrix.

    Args:
        mat: The matrix to check.

    Returns:
        Returns `True` if the matrix is square and `False` otherwise.

    """
    if len(mat.shape) != 2:
        raise ValueError("The variable is not a matrix.")
    return mat.shape[0] == mat.shape[1]
