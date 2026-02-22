"""Checks if the matrix is a unitary matrix."""

import numpy as np

from toqito.matrix_props import is_square


def is_unitary(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check if matrix is unitary [@WikiUniMat].

    A matrix is unitary if its inverse is equal to its conjugate transpose.

    Alternatively, a complex square matrix \(U\) is unitary if its conjugate transpose
    \(U^*\) is also its inverse, that is, if

    \[
        \begin{equation}
            U^* U = U U^* = \mathbb{I},
        \end{equation}
    \]

    where \(\mathbb{I}\) is the identity matrix.

    Examples:

    Consider the following matrix

    \[
        X = \begin{pmatrix}
            0 & 1 \\
            1 & 0
            \end{pmatrix}
    \]

    our function indicates that this is indeed a unitary matrix.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_props import is_unitary
    
    A = np.array([[0, 1], [1, 0]])
    
    print(is_unitary(A))
    ```

    We may also use the `random_unitary` function from `toqito`, and can verify that a randomly
    generated matrix is unitary

    ```python exec="1" source="above"
    from toqito.matrix_props import is_unitary
    from toqito.rand import random_unitary
    
    mat = random_unitary(2)
    
    print(is_unitary(mat))
    ```

    Alternatively, the following example matrix \(B\) defined as

    \[
        B = \begin{pmatrix}
            1 & 0 \\
            1 & 1
            \end{pmatrix}
    \]

    is not unitary.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_props import is_unitary
    
    B = np.array([[1, 0], [1, 1]])
    
    print(is_unitary(B))
    ```

    Args:
        mat: Matrix to check.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        Return `True` if matrix is unitary, and `False` otherwise.

    """
    if not is_square(mat):
        return False

    uc_u_mat = mat.conj().T @ mat
    u_uc_mat = mat @ mat.conj().T
    id_mat = np.eye(len(mat))

    # If U^* @ U = I U @ U^*, the matrix "U" is unitary.
    return bool(
        np.allclose(uc_u_mat, id_mat, rtol=rtol, atol=atol) and np.allclose(u_uc_mat, id_mat, rtol=rtol, atol=atol)
    )
