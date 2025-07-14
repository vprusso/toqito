"""Checks if the matrix is a positive semidefinite matrix."""

import numpy as np

from toqito.matrix_props import is_hermitian


def is_positive_semidefinite(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check if matrix is positive semidefinite (PSD) :footcite:`WikiPosDef`.

    Examples
    ==========

    Consider the following matrix

    .. math::
        A = \begin{pmatrix}
                1 & -1 \\
                -1 & 1
            \end{pmatrix}

    our function indicates that this is indeed a positive semidefinite matrix.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import is_positive_semidefinite

     A = np.array([[1, -1], [-1, 1]])

     is_positive_semidefinite(A)

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                -1 & -1 \\
                -1 & -1
            \end{pmatrix}

    is not positive semidefinite.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import is_positive_semidefinite

     B = np.array([[-1, -1], [-1, -1]])

     is_positive_semidefinite(B)

    References
    ==========
    .. footbibliography::



    :param mat: Matrix to check.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: Return :code:`True` if matrix is PSD, and :code:`False` otherwise.

    """
    if not is_hermitian(mat, rtol, atol):
        return False
    evals, _ = np.linalg.eigh(mat)
    return all(x >= -abs(atol) for x in evals)
