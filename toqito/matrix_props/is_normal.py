"""Checks if the matrix is a normal matrix."""

import numpy as np

from toqito.matrix_props import is_square


def is_normal(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Determine if a matrix is normal :footcite:`WikiNorm`.

    A matrix is normal if it commutes with its adjoint

    .. math::
        \begin{equation}
            [X, X^*] = 0,
        \end{equation}

    or, equivalently if

    .. math::
        \begin{equation}
            X^* X = X X^*
        \end{equation}.

    Examples
    ==========

    Consider the following matrix

    .. math::
        A = \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}

    our function indicates that this is indeed a normal matrix.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import is_normal

     A = np.identity(4)

     is_normal(A)

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                1 & 2 & 3 \\
                4 & 5 & 6 \\
                7 & 8 & 9
            \end{pmatrix}

    is not normal.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import is_normal

     B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

     is_normal(B)

    References
    ==========
    .. footbibliography::



    :param mat: The matrix to check.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: Returns :code:`True` if the matrix is normal and :code:`False` otherwise.

    """
    if not is_square(mat):
        return False
    return np.allclose(mat @ mat.conj().T, mat.conj().T @ mat, rtol=rtol, atol=atol)
