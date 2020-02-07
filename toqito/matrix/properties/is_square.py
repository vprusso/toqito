"""Determines whether or not a matrix is square."""
import numpy as np


def is_square(mat: np.ndarray) -> bool:
    r"""
    Determines if a matrix is square.

    A matrix is square if the dimensions of the rows and columns are equivalent.

    :param mat: The matrix to check.
    :return: Returns True if the matrix is square and False otherwise.
    """
    return mat.shape[0] == mat.shape[1]
