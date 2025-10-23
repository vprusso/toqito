"""Checks if the matrix is a positive definite matrix."""

import numpy as np


def is_positive_definite(mat: np.ndarray) -> bool:
    r"""Check if matrix is positive definite (PD) :footcite:`WikiPosDef`.

    Examples
    ==========

    Consider the following matrix

    .. math::
        A = \begin{pmatrix}
                2 & -1 & 0 \\
                -1 & 2 & -1 \\
                0 & -1 & 2
            \end{pmatrix}

    our function indicates that this is indeed a positive definite matrix.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import is_positive_definite

     A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])

     is_positive_definite(A)

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                -1 & -1 \\
                -1 & -1
            \end{pmatrix}

    is not positive definite.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import is_positive_definite

     B = np.array([[-1, -1], [-1, -1]])

     is_positive_definite(B)

    See Also
    ========
    :py:func:`~toqito.matrix_props.is_positive_semidefinite.is_positive_semidefinite`

    References
    ==========
    .. footbibliography::



    :param mat: Matrix to check.
    :return: Return :code:`True` if matrix is positive definite, and :code:`False` otherwise.

    """
    if np.array_equal(mat, mat.conj().T):
        try:
            # Cholesky decomp requires that the matrix in question is
            # positive-definite. It will throw an error if this is not the case
            # that we catch here.
            np.linalg.cholesky(mat)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False
