"""Performs the vec operation on a matrix."""

import numpy as np


def vec(mat: np.ndarray) -> np.ndarray:
    r"""Perform the vec operation on a matrix.

    For more info, see Section: The Operator-Vector Correspondence from [@Watrous_2018_TQI].

    The function reorders the given matrix into a column vector by stacking the columns of the matrix sequentially.

    For instance, for the following matrix:

    \[
        X =
        \begin{pmatrix}
            1 & 2 \\
            3 & 4
        \end{pmatrix}
    \]

    it holds that

    \[
        \text{vec}(X) = \begin{pmatrix} 1 & 3 & 2 & 4 \end{pmatrix}^T
    \]

    More formally, the vec operation is defined by

    \[
        \text{vec}(E_{a,b}) = e_a \otimes e_b
    \]

    for all \(a\) and \(b\) where

    \[
        E_{a,b}(c,d) = \begin{cases}
                          1 & \text{if} \ (c,d) = (a,b) \\
                          0 & \text{otherwise}
                        \end{cases}
    \]

    for all \(c\) and \(d\) and where

    \[
        e_a(b) = \begin{cases}
                     1 & \text{if} \ a = b \\
                     0 & \text{if} \ a \not= b
                 \end{cases}
    \]

    for all \(a\) and \(b\).

    Examples:

    Consider the following matrix

    \[
        A = \begin{pmatrix}
                1 & 2 \\
                3 & 4
            \end{pmatrix}
    \]

    Performing the \(\text{vec}\) operation on \(A\) yields

    \[
        \text{vec}(A) = \left[1, 3, 2, 4 \right]^{T}.
    \]

    ```python exec="1" source="above"
    import numpy as np
    from toqito.perms import vec
    
    X = np.array([[1, 2], [3, 4]])
    
    print(vec(X))
    ```

    !!! See Also
        [`unvec()`][toqito.matrix_ops.unvec.unvec]

    Args:
        mat: The input matrix.

    Returns:
        The vec representation of the matrix.

    """
    return mat.reshape((-1, 1), order="F")
