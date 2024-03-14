"""Is matrix circulant."""

import numpy as np


def is_circulant(mat: np.ndarray) -> bool:
    r"""Determine if matrix is circulant :cite:`WikiCirc`.

    A circulant matrix is a square matrix in which all row vectors are composed
    of the same elements and each row vector is rotated one element to the right
    relative to the preceding row vector.

    Examples
    ==========

    Consider the following matrix:

    .. math::
        C = \begin{pmatrix}
                4 & 1 & 2 & 3 \\
                3 & 4 & 1 & 2 \\
                2 & 3 & 4 & 1 \\
                1 & 2 & 3 & 4
            \end{pmatrix}

    As can be seen, this matrix is circulant. We can verify this in
    :code:`toqito` as

    >>> from toqito.matrix_props import is_circulant
    >>> import numpy as np
    >>> mat = np.array([[4, 1, 2, 3], [3, 4, 1, 2], [2, 3, 4, 1], [1, 2, 3, 4]])
    >>> is_circulant(mat)
    True

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param mat: Matrix to check the circulancy of.
    :return: Return `True` if :code:`mat` is circulant; `False` otherwise.

    """
    n, m = mat.shape
    if n != m:
        return False

    for i in range(n - 1):
        row = mat[i + 1]
        shifted = np.roll(mat[i], 1)
        if not np.allclose(row, shifted):
            return False
    return True
