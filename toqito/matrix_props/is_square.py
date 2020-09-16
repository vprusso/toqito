"""Is matrix a square matrix."""
import numpy as np


def is_square(mat: np.ndarray) -> bool:
    r"""
    Determine if a matrix is square [WikSquare]_.

    A matrix is square if the dimensions of the rows and columns are equivalent.

    Examples
    ==========

    Consider the following matrix

    .. math::
        A = \begin{pmatrix}
                1 & 2 & 3 \\
                4 & 5 & 6 \\
                7 & 8 & 9
            \end{pmatrix}

    our function indicates that this is indeed a square matrix.

    >>> from toqito.matrix_props import is_square
    >>> import numpy as np
    >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> is_square(A)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                1 & 2 & 3 \\
                4 & 5 & 6
            \end{pmatrix}

    is not square.

    >>> from toqito.matrix_props import is_square
    >>> import numpy as np
    >>> B = np.array([[1, 2, 3], [4, 5, 6]])
    >>> is_square(B)
    False

    References
    ==========
    .. [WikSquare] Wikipedia: Square matrix.
        https://en.wikipedia.org/wiki/Square_matrix

    :param mat: The matrix to check.
    :return: Returns :code:`True` if the matrix is square and :code:`False` otherwise.
    """
    if len(mat.shape) != 2:
        raise ValueError("The variable is not a matrix.")
    return mat.shape[0] == mat.shape[1]
