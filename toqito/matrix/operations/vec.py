"""Computes the vec representation of a given matrix."""
import numpy as np


def vec(mat: np.ndarray) -> np.ndarray:
    """
    Perform the vec operation on a matrix.

    Stacks the rows of the matrix on top of each other to
    obtain the "vec" representation of the matrix.

    :param mat: The input matrix.
    :return: The vec representation of the matrix.
    """
    return mat.reshape((-1, 1), order="F")
