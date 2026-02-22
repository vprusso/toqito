"""Checks if the matrix is a permutation matrix."""

import numpy as np


def is_permutation(mat: np.ndarray) -> bool:
    r"""Determine if a matrix is a permutation matrix [@WikiPerm].

    A matrix is a permutation matrix if each row and column has a
    single element of 1 and all others are 0.

    Examples:

    Consider the following permutation matrix

    \[
        A = \begin{pmatrix}
                1 & 0 & 0 \\
                0 & 0 & 1 \\
                0 & 1 & 0
            \end{pmatrix}
    \]

    which is indeed a permutation matrix.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_props import is_permutation
    
    A = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    
    print(is_permutation(A))
    ```


    Alternatively, the following example matrix \(B\) defined as

    \[
        B = \begin{pmatrix}
                1 & 0 & 0 \\
                1 & 0 & 0 \\
                1 & 0 & 0
            \end{pmatrix}
    \]

    has 2 columns with all zero values and is thus not a
    permutation matrix.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_props import is_permutation
    
    B = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    
    print(is_permutation(B))
    ```

    Args:
        mat: The matrix to check.

    Returns:
        Returns `True` if the matrix is a permutation matrix and `False` otherwise.

    """
    for i in np.nditer(mat):
        if i not in (0, 1):
            return False

    if all(sum(row) == 1 for row in mat):
        return all(sum(col) == 1 for col in zip(*mat))
    return False
