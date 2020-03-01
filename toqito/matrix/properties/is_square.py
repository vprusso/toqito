"""Determines whether or not a matrix is square."""
import numpy as np


def is_square(mat: np.ndarray) -> bool:
    r"""
    Determine if a matrix is square.

    A matrix is square if the dimensions of the rows and columns are equivalent.

    References:
    [1] Wikipedia: Square matrix.
        https://en.wikipedia.org/wiki/Square_matrix

    :param mat: The matrix to check.
    :return: Returns True if the matrix is square and False otherwise.
    """
    return mat.shape[0] == mat.shape[1]
