"""Checks if the matrix is a projection matrix."""

import numpy as np

from toqito.matrix_props import is_square


def is_projection(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check if matrix is a projection matrix :footcite:`WikiProjMat`.

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

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import is_projection

     A = np.array([[0, 1], [0, 1]])

     is_projection(A)

    A common use case in quantum mechanics is checking if a matrix represents a valid quantum measurement
    projector. For example, the projector onto the :math:`|0\rangle` state is defined as:

    .. math::
        P_0 = |0\rangle\langle 0| = \begin{pmatrix}
                1 & 0 \\
                0 & 0
              \end{pmatrix}

    This projector satisfies :math:`P_0^2 = P_0` and is positive semidefinite.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import is_projection

     # Projector onto |0⟩ state
     ket_0 = np.array([[1], [0]])
     proj_0 = ket_0 @ ket_0.conj().T

     is_projection(proj_0)

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                -1 & -1 \\
                -1 & -1
            \end{pmatrix}

    is not positive definite.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import is_projection

     B = np.array([[-1, -1], [-1, -1]])

     is_projection(B)


    References
    ==========
    .. footbibliography::



    :param mat: Matrix to check.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: Return :code:`True` if matrix is a projection matrix, and :code:`False` otherwise.

    """
    if not is_square(mat):
        return False
    return np.allclose(np.linalg.matrix_power(mat, 2), mat, rtol=rtol, atol=atol)
