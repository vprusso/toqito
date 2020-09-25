"""Is matrix the identity matrix."""
import numpy as np
from toqito.matrix_props import is_square


def is_identity(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-8) -> bool:
    r"""
    Check if matrix is the identity matrix [WikIdentity]_.

    For dimension :math:`n`, the :math:`n \times n` identity matrix is defined as

    .. math::
        I_n =
        \begin{pmatrix}
            1 & 0 & 0 & \ldots & 0 \\
            0 & 1 & 0 & \ldots & 0 \\
            0 & 0 & 1 & \ldots & 0 \\
            \vdots & \vdots & \vdots & \ddots & \vdots \\
            0 & 0 & 0 & \ldots & 1
        \end{pmatrix}.

    Examples
    ==========

    Consider the following matrix:

    .. math::
        A = \begin{pmatrix}
                1 & 0 & 0 \\
                0 & 1 & 0 \\
                0 & 0 & 1
            \end{pmatrix}

    our function indicates that this is indeed the identity matrix of dimension
    3.

    >>> from toqito.matrix_props import is_identity
    >>> import numpy as np
    >>> mat = np.eye(3)
    >>> is_identity(mat)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                1 & 2 & 3 \\
                4 & 5 & 6 \\
                7 & 8 & 9
            \end{pmatrix}

    is not an identity matrix.

    >>> from toqito.matrix_props import is_identity
    >>> import numpy as np
    >>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> is_identity(mat)
    False

    References
    ==========
    .. [WikIdentity] Wikipedia: Identity matrix
        https://en.wikipedia.org/wiki/Identity_matrix

    :param mat: Matrix to check.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: Return :code:`True` if matrix is the identity matrix, and
            :code:`False` otherwise.
    """
    if not is_square(mat):
        return False
    id_mat = np.eye(len(mat))
    return np.allclose(mat, id_mat, rtol=rtol, atol=atol)
