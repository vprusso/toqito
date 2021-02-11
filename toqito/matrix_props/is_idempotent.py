"""Is the matrix idempotent."""
import numpy as np
from toqito.matrix_props import is_square


def is_idempotent(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-8) -> bool:
    r"""
    Check if matrix is the idempotent matrix [WikIdempotent]_.

    An *idempotent matrix* is a square matrix, which, when multiplied by itself, yields itself.
    That is, the matrix :math:`A` is idempotent if and only if :math:`A^2 = A`.

    Examples
    ==========

    The following is an example of a :math:`2 x 2` idempotent matrix:

    .. math::
        A = \begin{pmatrix}
            3 & -6 \\
            1 & -2
        \end{pmatrix}

    >>> from toqito.matrix_props import is_idempotent
    >>> import numpy as np
    >>> mat = np.array([[3, -6], [1, -2]])
    >>> is_idempotent(mat)

    Alternatively, the following matrix

    .. math::
        B = \begin{pmatrix}
                1 & 2 & 3 \\
                4 & 5 & 6 \\
                7 & 8 & 9
            \end{pmatrix}

    is not idempotent.

    >>> from toqito.matrix_props import is_idempotent
    >>> import numpy as np
    >>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> is_idempotent(mat)
    False

    References
    ==========
    .. [WikIdempotent] Wikipedia: Idempotent matrix
        https://en.wikipedia.org/wiki/Idempotent_matrix

    :param mat: Matrix to check.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: Return :code:`True` if matrix is the idempotent matrix, and
            :code:`False` otherwise.
    """
    if not is_square(mat):
        return False
    return np.allclose(mat, mat @ mat, rtol=rtol, atol=atol)
