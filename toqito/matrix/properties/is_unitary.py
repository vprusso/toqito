"""Determines whether or not a matrix is unitary."""
from typing import Union
import numpy as np


def is_unitary(mat: Union[np.ndarray, np.matrix]) -> bool:
    """
    Check if matrix is unitary.

    A matrix is unitary if its inverse is equal to its conjugate transose.

    :param mat: Matrix to check.
    :return: Return `True` if matrix is unitary, and `False` otherwise.
    """
    if not isinstance(mat, np.matrix):
        mat = np.matrix(mat)

    # If U^* * U = I U * U^*, the matrix "U" is unitary.
    return np.allclose(np.identity(mat.shape[0]), mat * mat.H)
