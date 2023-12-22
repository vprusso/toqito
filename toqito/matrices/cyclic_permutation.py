"""Generate a cyclic permutation matrix."""

import numpy as np


def cyclic_permutation_matrix(n: int) -> np.ndarray:
    r"""Create the cyclic permutation matrix P for a given dimension n.

    This function creates a cyclic permutation matrix which is a special type of square matrix
    that represents a cyclic permutation of its rows.

    Examples
    ==========
    Generate various cyclic permutation matrices.

    >>> from this_module import cyclic_permutation_matrix
    >>> n = 4
    >>> cyclic_permutation_matrix = cyclic_permutation_matrix(n)
    >>> print(cyclic_permutation_matrix)
    [[0 0 0 1]
    [1 0 0 0]
    [0 1 0 0]
    [0 0 1 0]]

    :param n: int
        The number of rows and columns in the cyclic permutation matrix.

    :return: numpy.ndarray
        A NumPy array representing a cyclic permutation matrix of dimension `n x n`.
        Each row of the matrix is shifted one position to the right in a cyclic manner,
        creating a circular permutation pattern.
    """
    p_mat = np.zeros((n, n), dtype=int)
    np.fill_diagonal(p_mat[1:], 1)
    p_mat[0, -1] = 1
    return p_mat
