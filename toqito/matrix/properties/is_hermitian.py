"""Determines whether or not a matrix is Hermitian."""
import numpy as np


def is_hermitian(mat: np.ndarray) -> bool:
    """
    Check if matrix is Hermitian.

    :param mat: Matrix to check.
    :return: Return True if matrix is Hermitian, and False otherwise.
    """
    return np.allclose(mat, mat.conj().T)
