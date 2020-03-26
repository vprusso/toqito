"""Determines whether or not a matrix is unitary."""
from typing import Union
import numpy as np


def is_unitary(mat: Union[np.ndarray, np.matrix]) -> bool:
    r"""
    Check if matrix is unitary.

    A matrix is unitary if its inverse is equal to its conjugate transpose.

    Alternatively, a complex square matrix :math:`U` is unitary if its conjugate
    transpose :math:`U^*` is also its inverse, that is, if

    .. math::
        \begin{equation}
            U^* U = U U^* = I,
        \end{equation}

    where :math:`I` is the identity matrix.

    References:
        [1] Wikipedia: Unitary matrix.
        https://en.wikipedia.org/wiki/Unitary_matrix

    :param mat: Matrix to check.
    :return: Return `True` if matrix is unitary, and `False` otherwise.
    """
    if not isinstance(mat, np.matrix):
        mat = np.matrix(mat)

    # If U^* * U = I U * U^*, the matrix "U" is unitary.
    return np.allclose(np.identity(mat.shape[0]), mat * mat.conj().T)
