"""Checks if the matrix is a square matrix."""

import numpy as np


def is_square(mat: np.ndarray) -> bool:
    r"""Determine if a matrix is square :footcite:`WikiSqMat`.

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

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import is_square

     A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

     is_square(A)

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                1 & 2 & 3 \\
                4 & 5 & 6
            \end{pmatrix}

    is not square.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import is_square

     B = np.array([[1, 2, 3], [4, 5, 6]])

     is_square(B)


    References
    ==========
    .. footbibliography::



    :raises ValueError: If variable is not a matrix.
    :param mat: The matrix to check.
    :return: Returns :code:`True` if the matrix is square and :code:`False` otherwise.

    """
    if len(mat.shape) != 2:
        raise ValueError("The variable is not a matrix.")
    return mat.shape[0] == mat.shape[1]
