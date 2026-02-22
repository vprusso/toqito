"""Checks if matrix is pseudo unitary."""

import numpy as np

from toqito.matrix_props import is_square


def is_pseudo_unitary(mat: np.ndarray, p: int, q: int, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check if a matrix is pseudo-unitary.

    A matrix A of size (p+q)x(p+q) is pseudo-unitary with respect to a given signature matrix J if it satisfies

    \[
        A^* J A = J,
    \]

    where:

    - \(A^*\) is the conjugate transpose (Hermitian transpose) of \(A\),
    - \(J\) is a diagonal matrix with first \(p\) diagonal entries equal to 1 and next \(q\) diagonal entries equal to -1

    Examples:

    Consider the following matrix:

    \[
        A = \begin{pmatrix}
            cosh(1) & sinh(1) \\
            sinh(1) & cosh(1)
        \end{pmatrix}
    \]

    with the signature matrix:

    \[
        J = \begin{pmatrix}
            1 & 0 \\
            0 & -1
        \end{pmatrix}
    \]

    Our function confirms that \(A\) is pseudo-unitary.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_props import is_pseudo_unitary
    
    A = np.array([[np.cosh(1), np.sinh(1)], [np.sinh(1), np.cosh(1)]])
    
    print(is_pseudo_unitary(A, p=1, q=1))
    ```


    However, the following matrix :math:B

    \[
        B = \begin{pmatrix}
            1 & 0 \\
            1 & 1
        \end{pmatrix}
    \]

    is not pseudo-unitary with respect to the same signature matrix:

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_props import is_pseudo_unitary
    
    B = np.array([[1, 0], [1, 1]])
    
    print(is_pseudo_unitary(B, p=1, q=1))
    ```

    Raises:
        ValueError: When p < 0 or q < 0.

    Args:
        mat: The matrix to check.
        p: Number of positive entries in the signature matrix.
        q: Number of negative entries in the signature matrix.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        Return :code:True if the matrix is pseudo-unitary, and :code:False otherwise.

    """
    if p < 0 or q < 0:
        raise ValueError("p and q must be non-negative")

    if not is_square(mat):
        return False

    if p + q != mat.shape[0]:
        return False

    signature = np.diag(np.hstack((np.ones(p), -np.ones(q))))
    ac_j_a_mat = mat.conj().T @ signature @ mat

    return bool(np.allclose(ac_j_a_mat, signature, rtol=rtol, atol=atol))
