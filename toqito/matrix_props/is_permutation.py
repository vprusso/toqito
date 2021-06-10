"""Is matrix a permutation matrix."""
import numpy as np
from toqito.matrix_props import is_square

def is_permutation(mat: np.ndarray) -> bool:
    r"""
    Determine if a matrix is a permutation matrix [WikiPermutation]_.

    A matrix is a permutation matrix if each row and column has a
    single element of 1 and all others are 0.

    Examples
    ==========

    Consider the following matrix

    .. math::
        A = \begin{pmatrix}
                1 & 2 & 3 \\
                4 & 5 & 6 \\
                7 & 8 & 9
            \end{pmatrix}

    our function indicates that this is indeed a permutation matrix.

    >>> from toqito.matrix_props import is_permutation
    >>> import numpy as np
    >>> A = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    >>> is_permutation(A)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                1 & 2 & 3 \\
                4 & 5 & 6 \\
                7 & 8 & 9
            \end{pmatrix}

    >>> from toqito.matrix_props import is_permutation
    >>> import numpy as np
    >>> B = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    >>> is_permutation(B)
    False

    References
    ==========
    .. [WikiPermutation] Wikipedia: Permutation matrix.
        https://en.wikipedia.org/wiki/Permutation_matrix

    :param mat: The matrix to check.
    :return: Returns :code:`True` if the matrix is a permutation matrix and :code:`False` otherwise.
    """
    if not is_square(mat):
        return False

    """ each element in set of (0,1) check """
    for i in np.nditer(mat):
        if (i not in (0, 1)):
            return False

    """ check sum of each individual row and column to be 1 """
    if all(sum(row) == 1 for row in mat):
        return all(sum(col) == 1 for col in zip(*mat))
    return False
