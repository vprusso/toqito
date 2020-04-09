"""Determines whether or not a matrix is square."""
import numpy as np


def is_square(mat: np.ndarray) -> bool:
    r"""
    Determine if a matrix is square [9]_.

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

    >>> from toqito.matrix.properties.is_square import is_square
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

    >>> from toqito.matrix.properties.is_square import is_square
    >>> import numpy as np
    >>> B = np.array([[1, 2, 3], [4, 5, 6]])
    >>> is_square(B)
    False

    References
    ==========
    .. [9] Wikipedia: Square matrix.
        https://en.wikipedia.org/wiki/Square_matrix

    :param mat: The matrix to check.
    :return: Returns True if the matrix is square and False otherwise.
    """
    return mat.shape[0] == mat.shape[1]
