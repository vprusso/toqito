"""Is matrix a symmetric matrix."""
import numpy as np

from toqito.matrix_props import is_square


def is_symmetric(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""
    Determine if a matrix is symmetric [WikSym]_.

    The following 3x3 matrix is an example of a symmetric matrix:

    .. math::

        \begin{pmatrix}
            1 & 7 & 3 \\
            7 & 4 & -5 \\
            3 &-5 & 6
        \end{pmatrix}

    Examples
    ==========

    Consider the following matrix

    .. math::
        A = \begin{pmatrix}
                1 & 7 & 3 \\
                7 & 4 & -5 \\
                3 & -5 & 6
            \end{pmatrix}

    our function indicates that this is indeed a symmetric matrix.

    >>> from toqito.matrix_props import is_symmetric
    >>> import numpy as np
    >>> A = np.array([[1, 7, 3], [7, 4, -5], [3, -5, 6]])
    >>> is_symmetric(A)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                1 & 2 \\
                4 & 5
            \end{pmatrix}

    is not symmetric.

    >>> from toqito.matrix_props import is_symmetric
    >>> import numpy as np
    >>> B = np.array([[1, 2], [3, 4]])
    >>> is_symmetric(B)
    False

    References
    ==========
    .. [WikSym] Wikipedia: Symmetric matrix
        https://en.wikipedia.org/wiki/Symmetric_matrix

    :param mat: The matrix to check.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: Returns :code:`True` if the matrix is symmetric and :code:`False` otherwise.
    """
    if not is_square(mat):
        return False
    return np.allclose(mat, mat.T, rtol=rtol, atol=atol)
