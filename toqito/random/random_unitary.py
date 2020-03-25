"""Generates a random unitary or orthogonal matrix."""
from typing import List, Union
import numpy as np


def random_unitary(dim: Union[List[int], int], is_real: bool = False) -> np.ndarray:
    """
    Generate a random unitary or orthogonal matrix.

    Calculates a random unitary matrix (if `is_real = False`) or a random real
    orthogonal matrix (if `is_real = True`), uniformly distributed according to
    the Haar measure.

    [1] References:
    How to generate a random unitary matrix,
    Maris Ozols
    March 16, 2009,
    http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20%5Bpaper%5D.pdf

    :param dim: The number of rows (and columns) of the unitary matrix.
    :param is_real: Boolean denoting whether the returned matrix has real
                    entries or not. Default is `False`.
    :return: A `dim`-by-`dim` random unitary matrix.
    """
    if isinstance(dim, int):
        dim = [dim, dim]

    # Construct the Ginibre ensemble.
    gin = np.random.rand(dim[0], dim[1])

    if not is_real:
        gin = gin + 1j * np.random.rand(dim[0], dim[1])

    # QR decomposition of the Ginibre ensemble.
    q_mat, r_mat = np.linalg.qr(gin)

    # Compute U from QR decomposition.
    r_mat = np.sign(np.diag(r_mat))

    # Protect against potentially zero diagonal entries.
    r_mat[r_mat == 0] = 1

    return np.matmul(q_mat, np.diag(r_mat))
