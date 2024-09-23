"""Is matrix an anti-Hermitian matrix."""

import numpy as np

from toqito.matrix_props.is_hermitian import is_hermitian


def is_anti_hermitian(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check if matrix is anti-Hermitian (a.k.a. skew-Hermitian) :cite:`WikiAntiHerm`.

    An anti-Hermitian matrix is a complex square matrix that is equal to the negative of its own
    conjugate transpose.

    Examples
    ==========

    Consider the following matrix:

    .. math::
        A = \begin{pmatrix}
                2j & -1 + 2j & 4j \\
                1 + 2j & 3j & -1 \\
                4j & 1 & 1j
            \end{pmatrix}

    our function indicates that this is indeed an anti-Hermitian matrix as it holds that

    .. math::
        A = -A^*.

    >>> from toqito.matrix_props import is_anti_hermitian
    >>> import numpy as np
    >>> mat = np.array([[2j, -1 + 2j, 4j], [1 + 2j, 3j, -1], [4j, 1, 1j]])
    >>> is_anti_hermitian(mat)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                1 & 2 & 3 \\
                4 & 5 & 6 \\
                7 & 8 & 9
            \end{pmatrix}

    is not anti-Hermitian.

    >>> from toqito.matrix_props import is_anti_hermitian
    >>> import numpy as np
    >>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> is_anti_hermitian(mat)
    False

    References
    ==========
    .. bibliography::
        :filter: docname in docnames



    :param mat: Matrix to check.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: Return True if matrix is anti-Hermitian, and False otherwise.

    """
    return is_hermitian(mat * 1j, rtol, atol)
