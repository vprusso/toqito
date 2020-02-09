"""Determines whether or not a matrix is positive definite."""
import numpy as np
from toqito.matrix.properties.is_square import is_square


def is_pd(mat: np.ndarray) -> bool:
    """
    Check if matrix is positive definite (PD).

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
