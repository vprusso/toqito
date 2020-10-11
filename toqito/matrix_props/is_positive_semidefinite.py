"""Is matrix a positive semidefinite matrix."""
import numpy as np

from toqito.matrix_props import is_hermitian


def is_positive_semidefinite(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
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

    >>> from toqito.matrix_props import is_positive_semidefinite
    >>> import numpy as np
    >>> A = np.array([[1, -1], [-1, 1]])
    >>> is_positive_semidefinite(A)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                -1 & -1 \\
                -1 & -1
            \end{pmatrix}

    is not positive semidefinite.

    >>> from toqito.matrix_props import is_positive_semidefinite
    >>> import numpy as np
    >>> B = np.array([[-1, -1], [-1, -1]])
    >>> is_positive_semidefinite(B)
    False

    References
    ==========
    .. [WikPSD] Wikipedia: Definiteness of a matrix.
        https://en.wikipedia.org/wiki/Definiteness_of_a_matrix

    :param mat: Matrix to check.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: Return :code:`True` if matrix is PSD, and :code:`False` otherwise.
    """
    if not is_hermitian(mat, rtol, atol):
        return False
    evals, _ = np.linalg.eigh(mat)
    return all(x >= -abs(atol) for x in evals)
