"""Determines whether or not a matrix is diagonal."""
import numpy as np
from toqito.matrix.properties.is_square import is_square


def is_diagonal(mat: np.ndarray) -> bool:
    r"""
    Determines if a matrix is diagonal.

    A matrix is diagonal if the matrix is square and if the diagonal of the
    matrix is non-zero, while the off-diagonal elements are all zero.

    :param mat: The matrix to check.
    :return: Returns True if the matrix is diagonal and False otherwise.
    """
    if not is_square(mat):
        return False
    return np.count_nonzero(mat - np.diag(np.diagonal(mat))) == 0
