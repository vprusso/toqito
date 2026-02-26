"""Checks if the matrix is commuting."""

import numpy as np


def is_commuting(mat_1: np.ndarray, mat_2: np.ndarray) -> bool:
    r"""Determine if two linear operators commute with each other [@WikiComm].

    For any pair of operators \(X, Y \in \text{L}(\mathcal{X})\), the
    Lie bracket \(\left[X, Y\right] \in \text{L}(\mathcal{X})\) is defined
    as

    \[
        \left[X, Y\right] = XY - YX.
    \]

    It holds that \(\left[X,Y\right]=0\) if and only if \(X\) and
    \(Y\) commute (Section: Lie Brackets And Commutants from [@Watrous_2018_TQI]).

    Examples:
    Consider the following matrices:

    \[
        A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix},
        \quad \text{and} \quad
        B = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}.
    \]

    It holds that \(AB=0\), however

    \[
        BA = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix} = A,
    \]

    and hence, do not commute.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_props import is_commuting

    mat_1 = np.array([[0, 1], [0, 0]])
    mat_2 = np.array([[1, 0], [0, 0]])

    print(is_commuting(mat_1, mat_2))
    ```

    Consider the following pair of matrices:

    \[
        A = \begin{pmatrix}
            1 & 0 & 0 \\
            0 & 1 & 0 \\
            1 & 0 & 2
            \end{pmatrix} \quad \text{and} \quad
        B = \begin{pmatrix}
            2 & 4 & 0 \\
            3 & 1 & 0 \\
            -1 & -4 & 1
            \end{pmatrix}.
    \]

    It may be verified that \(AB = BA = 0\), and therefore \(A\) and
    \(B\) commute.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_props import is_commuting

    mat_1 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 2]])
    mat_2 = np.array([[2, 4, 0], [3, 1, 0], [-1, -4, 1]])

    print(is_commuting(mat_1, mat_2))
    ```

    Args:
        mat_1: First matrix to check.
        mat_2: Second matrix to check.

    Returns:
        Return `True` if `mat_1` commutes with `mat_2` and False otherwise.

    """
    return np.allclose(mat_1 @ mat_2 - mat_2 @ mat_1, 0)
