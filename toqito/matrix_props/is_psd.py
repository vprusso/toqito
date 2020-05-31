"""Is matrix a positive semidefinite matrix."""
import numpy as np

from toqito.matrix_props import is_square


def is_psd(mat: np.ndarray, tol: float = 1e-8) -> bool:
    r"""
    Check if matrix is positive semidefinite (PSD) [WikPSD]_.

    Examples
    ==========

    Consider the following matrix

    .. math::
        A = \begin{pmatrix}
                1 & -1 \\
                -1 & 1
            \end{pmatrix}

    our function indicates that this is indeed a positive semidefinite matrix.

    >>> from toqito.matrix_props import is_psd
    >>> import numpy as np
    >>> A = np.array([[1, -1], [-1, 1]])
    >>> is_psd(A)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                -1 & -1 \\
                -1 & -1
            \end{pmatrix}

    is not positive semidefinite.

    >>> from toqito.matrix_props import is_psd
    >>> import numpy as np
    >>> B = np.array([[-1, -1], [-1, -1]])
    >>> is_psd(B)
    False

    References
    ==========
    .. [WikPSD] Wikipedia: Definiteness of a matrix.
        https://en.wikipedia.org/wiki/Definiteness_of_a_matrix

    :param mat: Matrix to check.
    :param tol: Tolerance for numerical accuracy.
    :return: Return True if matrix is PSD, and False otherwise.
    """
    if not is_square(mat):
        return False
    return np.all(np.linalg.eigvalsh(mat) > -tol)
