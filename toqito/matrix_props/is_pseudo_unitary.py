"""Checks if matrix is pseudo unitary."""

import numpy as np

from toqito.matrix_props import is_square


def is_pseudo_unitary(mat: np.ndarray, p: int, q: int, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check if a matrix is pseudo-unitary.

    A matrix A of size (p+q)x(p+q) is pseudo-unitary with respect to a given signature matrix J if it satisfies

    .. math::
        A^* J A = J,

    where:
        - :math:A^* is the conjugate transpose (Hermitian transpose) of :math:A,
        - :math:J is a diagonal matrix with first p diagonal matrix equal to 1 and next q diagonal entries equal to -1

    Examples
    ==========

    Consider the following matrix:

    .. math::
        A = \begin{pmatrix}
            cosh(1) & sinh(1) \\
            sinh(1) & cosh(1)
        \end{pmatrix}

    with the signature matrix:

    .. math::
        J = \begin{pmatrix}
            1 & 0 \\
            0 & -1
        \end{pmatrix}

    Our function confirms that :math:`A` is pseudo-unitary.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import is_pseudo_unitary

     A = np.array([[np.cosh(1), np.sinh(1)], [np.sinh(1), np.cosh(1)]])

     is_pseudo_unitary(A, p=1, q=1)


    However, the following matrix :math:B

    .. math::
        B = \begin{pmatrix}
            1 & 0 \\
            1 & 1
        \end{pmatrix}

    is not pseudo-unitary with respect to the same signature matrix:

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import is_pseudo_unitary

     B = np.array([[1, 0], [1, 1]])

     is_pseudo_unitary(B, p=1, q=1)

    References
    ==========

    .. footbibliography::


    :param mat: The matrix to check.
    :param p: Number of positive entries in the signature matrix.
    :param q: Number of negative entries in the signature matrix.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :raises ValueError: When p < 0 or q < 0.
    :return: Return :code:True if the matrix is pseudo-unitary, and :code:False otherwise.

    """
    if p < 0 or q < 0:
        raise ValueError("p and q must be non-negative")

    if not is_square(mat):
        return False

    if p + q != mat.shape[0]:
        return False

    signature = np.diag(np.hstack((np.ones(p), -np.ones(q))))
    ac_j_a_mat = mat.conj().T @ signature @ mat

    return bool(np.allclose(ac_j_a_mat, signature, rtol=rtol, atol=atol))
