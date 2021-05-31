"""Is matrix a permutation matrix."""
import numpy as np

def is_permutation(mat: np.ndarray) -> bool:
    r"""
    Determine if a matrix is a permutation matrix [WikSquare]_.

    A matrix is a permutation matrix if each row and column has a single element of 1 and all others 0.

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

    >>> from toqito.matrix_props import is_permutation
    >>> import numpy as np
    >>> A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> is_permutation(A)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                1 & 2 & 3 \\
                4 & 5 & 6
                7 & 8 & 9
            \end{pmatrix}

    >>> from toqito.matrix_props import is_permutation
    >>> import numpy as np
    >>> B = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]] )
    >>> is_permutation(B)
    False

    References
    ==========
    .. [WikSquare] Wikipedia: Pemutation matrix.
        https://en.wikipedia.org/wiki/Permutation_matrix

    :param mat: The matrix to check.
    :return: Returns :code:`True` if the matrix is permutation and :code:`False` otherwise.
    """
    if len(mat.shape) != 2:
        raise ValueError("The variable is not a 2-D matrix.")

    if not is_square(mat):
        return False

    for i in mat:
      for j in i:
        if (j!=0)&(j!=1):
           return False

    for rowTotal in np.sum(mat,axis=1) :
        if rowTotal != 1 :
            return False
    for colTotal in np.sum(mat,axis=0) :
        if colTotal != 1 :
            return False
    return True

