"""Determines whether or not a matrix is diagonal."""
import numpy as np
from toqito.matrix.properties.is_square import is_square


def is_diagonal(mat: np.ndarray) -> bool:
    r"""
    Determine if a matrix is diagonal.

    A matrix is diagonal if the matrix is square and if the diagonal of the
    matrix is non-zero, while the off-diagonal elements are all zero.

    The following is an example of a 3-by-3 diagonal matrix:

    .. math::
        \begin{equation}
            \begin{pmatrix}
                1 & 0 & 0 \\
                0 & 2 & 0 \\
                0 & 0 & 3
            \end{pmatrix}
        \end{equation}

    This quick implementation is given by Daniel F. from StackOverflow in [2]:

    References:
        [1] Wikipedia: Diagonal matrix
        https://en.wikipedia.org/wiki/Diagonal_matrix

        [2] StackOverflow post
        https://stackoverflow.com/questions/43884189/

    :param mat: The matrix to check.
    :return: Returns True if the matrix is diagonal and False otherwise.
    """
    if not is_square(mat):
        return False
    i, j = mat.shape
    test = mat.reshape(-1)[:-1].reshape(i - 1, j + 1)
    return ~np.any(test[:, 1:])
