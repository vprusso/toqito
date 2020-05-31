"""Is matrix a projection matrix."""
import numpy as np
from toqito.matrix_props import is_psd


def is_projection(mat: np.ndarray) -> bool:
    r"""
    Check if matrix is a projection matrix [WikProj]_.

    A matrix is a projection matrix if it is positive semidefinite (PSD) and if

    .. math::
        \begin{equation}
            X^2 = X
        \end{equation}

    where :math:`X` is the matrix in question.

    Examples
    ==========

    Consider the following matrix

    .. math::
        A = \begin{pmatrix}
                0 & 1 \\
                0 & 1
            \end{pmatrix}

    our function indicates that this is indeed a projection matrix.

    >>> from toqito.matrix_props import is_projection
    >>> import numpy as np
    >>> A = np.array([[0, 1], [0, 1]])
    >>> is_projection(A)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                -1 & -1 \\
                -1 & -1
            \end{pmatrix}

    is not positive definite.

    >>> from toqito.matrix_props import is_projection
    >>> import numpy as np
    >>> B = np.array([[-1, -1], [-1, -1]])
    >>> is_projection(B)
    False

    References
    ==========
    .. [WikProj] Wikipedia: Projection matrix.
        https://en.wikipedia.org/wiki/Projection_matrix

    :param mat: Matrix to check.
    :return: Return True if matrix is a projection matrix, and False otherwise.
    """
    if not is_psd(mat):
        return False
    return np.allclose(np.linalg.matrix_power(mat, 2), mat)
