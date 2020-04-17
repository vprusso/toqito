"""Determines whether or not a matrix is unitary."""
from typing import Union
import numpy as np


def is_unitary(mat: Union[np.ndarray, np.matrix]) -> bool:
    r"""
    Check if matrix is unitary [11]_.

    A matrix is unitary if its inverse is equal to its conjugate transpose.

    Alternatively, a complex square matrix :math:`U` is unitary if its conjugate
    transpose :math:`U^*` is also its inverse, that is, if

    .. math::
        \begin{equation}
            U^* U = U U^* = \mathbb{I},
        \end{equation}

    where :math:`\mathbb{I}` is the identity matrix.

    Examples
    ==========

    Consider the following matrix

    .. math::
        X = \begin{pmatrix}
                                0 & 1 \\
                                1 & 0
                           \end{pmatrix}

    our function indicates that this is indeed a unitary matrix.

    >>> from toqito.linear_algebra.properties.is_unitary import is_unitary
    >>> import numpy as np
    >>> A = np.array([[0, 1], [1, 0]])
    >>> is_unitary(A)
    True

    We may also use the `random_unitary` function from `toqito`, and can verify
    that a randomly generated matrix is unitary

    >>> from toqito.linear_algebra.properties.is_unitary import is_unitary
    >>> from toqito.random.random_unitary import random_unitary
    >>> mat = random_unitary(2)
    >>> is_unitary(mat)
    True

    Alternatively, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                                1 & 0 \\
                                1 & 1
                             \end{pmatrix}

    is not unitary.

    >>> from toqito.linear_algebra.properties.is_unitary import is_unitary
    >>> import numpy as np
    >>> B = np.array([[1, 0], [1, 1]])
    >>> is_unitary(B)
    False

    References
    ==========
    .. [11] Wikipedia: Unitary matrix.
        https://en.wikipedia.org/wiki/Unitary_matrix

    :param mat: Matrix to check.
    :return: Return `True` if matrix is unitary, and `False` otherwise.
    """
    if not isinstance(mat, np.matrix):
        mat = np.matrix(mat)

    # If U^* * U = I U * U^*, the matrix "U" is unitary.
    return np.allclose(np.identity(mat.shape[0]), mat * mat.conj().T)
