"""Is matrix a positive definite matrix."""
import numpy as np


def is_pd(mat: np.ndarray) -> bool:
    r"""
    Check if matrix is positive definite (PD) [WikPD]_.

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

    >>> from toqito.matrix_props import is_pd
    >>> import numpy as np
    >>> A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    >>> is_pd(A)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                -1 & -1 \\
                -1 & -1
            \end{pmatrix}

    is not positive definite.

    >>> from toqito.matrix_props import is_pd
    >>> import numpy as np
    >>> B = np.array([[-1, -1], [-1, -1]])
    >>> is_pd(B)
    False

    See Also
    ========
    is_psd

    References
    ==========
    .. [WikPD] Wikipedia: Definiteness of a matrix.
        https://en.wikipedia.org/wiki/Definiteness_of_a_matrix

    :param mat: Matrix to check.
    :return: Return True if matrix is PD, and False otherwise.
    """
    if np.array_equal(mat, mat.T):
        try:
            np.linalg.cholesky(mat)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False
