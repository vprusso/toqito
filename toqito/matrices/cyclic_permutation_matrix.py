"""Generates a cyclic permutation matrix."""

import numpy as np


def cyclic_permutation_matrix(n: int, k: int = 1) -> np.ndarray:
    r"""Create the cyclic permutation matrix for a given dimension :code:`n` :footcite:`WikiCyclicPermutation`.

    This function creates a cyclic permutation matrix of 0's and 1's which is a special type of square matrix
    that represents a cyclic permutation of its rows. The function allows fixed points and successive applications.

    Examples
    ==========
    Generate fixed point.

    .. jupyter-execute::

     from toqito.matrices import cyclic_permutation_matrix

     cyclic_permutation_matrix(n=4)

    Generate successive application.

    .. jupyter-execute::

     from toqito.matrices import cyclic_permutation_matrix

     cyclic_permutation_matrix(n=4, k=3)

    References
    ==========
    .. footbibliography::


    :param n: int
        The number of rows and columns in the cyclic permutation matrix.

    :param k: int
        The power to which the elements are raised, representing successive applications.

    :return:
        A NumPy array representing a cyclic permutation matrix of dimension :code:`n x n`.
        Each row of the matrix is shifted one position to the right in a cyclic manner,
        creating a circular permutation pattern. If :code:`k` is specified, the function
        raises the matrix to the power of :code:`k`, representing successive applications
        of the cyclic permutation.

    """
    p_mat = np.zeros((n, n), dtype=int)
    np.fill_diagonal(p_mat[1:], 1)
    p_mat[0, -1] = 1

    result_mat = np.linalg.matrix_power(p_mat, k)
    return result_mat
