"""Determines whether or not a matrix is positive semidefinite."""
import numpy as np
from toqito.matrix.properties.is_square import is_square


def is_psd(mat: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check if matrix is positive semidefinite (PSD).

    References:
        [1] Wikipedia: Definiteness of a matrix.
        https://en.wikipedia.org/wiki/Definiteness_of_a_matrix

    :param mat: Matrix to check.
    :param tol: Tolerance for numerical accuracy.
    :return: Return True if matrix is PSD, and False otherwise.
    """
    if not is_square(mat):
        return False
    return np.all(np.linalg.eigvalsh(mat) > -tol)
