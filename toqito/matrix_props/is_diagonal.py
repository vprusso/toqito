"""Checks if the matrix is a diagonal matrix."""

import numpy as np

from toqito.matrix_props import is_square


def is_diagonal(mat: np.ndarray) -> bool:
    r"""Determine if a matrix is diagonal :cite:`WikiDiag`.

    A matrix is diagonal if the matrix is square and if the diagonal of the matrix is non-zero,
    while the off-diagonal elements are all zero.

    The following is an example of a 3-by-3 diagonal matrix:

    .. math::
        \begin{equation}
            \begin{pmatrix}
                1 & 0 & 0 \\
                0 & 2 & 0 \\
                0 & 0 & 3
            \end{pmatrix}
        \end{equation}

    This quick implementation is given by Daniel F. from StackOverflow in :cite:`SO_43884189`.

    Examples
    ==========

    Consider the following diagonal matrix:

    .. math::
        A = \begin{pmatrix}
                1 & 0 \\
                0 & 1
            \end{pmatrix}.

    Our function indicates that this is indeed a diagonal matrix:

    >>> from toqito.matrix_props import is_diagonal
    >>> import numpy as np
    >>> A = np.array([[1, 0], [0, 1]])
    >>> is_diagonal(A)
    np.True_

    Alternatively, the following example matrix

    .. math::
        B = \begin{pmatrix}
                1 & 2 \\
                3 & 4
            \end{pmatrix}

    is not diagonal, as shown using :code:`toqito`.

    >>> from toqito.matrix_props import is_diagonal
    >>> import numpy as np
    >>> B = np.array([[1, 2], [3, 4]])
    >>> is_diagonal(B)
    np.False_

    References
    ==========
    .. bibliography::
        :filter: docname in docnames



    :param mat: The matrix to check.
    :return: Returns :code:`True` if the matrix is diagonal
             and :code:`False` otherwise.

    """
    if not is_square(mat):
        return False
    i, j = mat.shape
    test = mat.reshape(-1)[:-1].reshape(i - 1, j + 1)
    return ~np.any(test[:, 1:])
