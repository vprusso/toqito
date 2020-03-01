"""Determines whether or not a matrix is a projection matrix."""
import numpy as np
from toqito.matrix.properties.is_psd import is_psd


def is_projection(mat: np.ndarray) -> bool:
    r"""
    Check if matrix is a projection matrix.

    A matrix is a projection matrix if it is positive semidefinite (PSD) and if

    :math: `X^2 = X`

    where `X` is the matrix in question.

    References:
    [1] Wikipedia: Projection matrix.
        https://en.wikipedia.org/wiki/Projection_matrix

    :param mat: Matrix to check.
    :return: Return True if matrix is a projection matrix, and False otherwise.
    """
    if not is_psd(mat):
        return False
    return np.allclose(np.linalg.matrix_power(mat, 2), mat)
