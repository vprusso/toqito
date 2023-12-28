"""Generate a cyclic permutation matrix."""

import numpy as np


def cyclic_permutation_matrix(n: int) -> np.ndarray:
    r"""Create the cyclic permutation matrix for a given dimension :code:`n` :cite:`WikiCyclicPermutation`.

    This function creates a cyclic permutation matrix which is a special type of square matrix
    that represents a cyclic permutation of its rows.

    The permutation can be written in cycle notation and two-line notation as:

    .. math::
        \begin{align*}
        \begin{aligned}
        &\begin{matrix}
        (1 & 4 & 6 & 8 & 3 & 7)(2)(5)
        \end{matrix}\\
        &= \begin{pmatrix}
        1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 \\
        4 & 2 & 7 & 6 & 5 & 8 & 1 & 3
        \end{pmatrix}\\
        &= \begin{pmatrix}
        1 & 4 & 6 & 8 & 3 & 7 & 2 & 5 \\
        4 & 6 & 8 & 3 & 7 & 1 & 2 & 5
        \end{pmatrix}
        \end{aligned}
        \end{align*}

    Examples
    ==========
    Generate various cyclic permutation matrices.

    >>> from this_module import cyclic_permutation_matrix
    >>> n = 4
    >>> cyclic_permutation_matrix = cyclic_permutation_matrix(n)
    >>> print(cyclic_permutation_matrix)
    ([[0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]])
    ...

    :param n: int
        The number of rows and columns in the cyclic permutation matrix.

    :return:
        A NumPy array representing a cyclic permutation matrix of dimension :code:`n x n`.
        Each row of the matrix is shifted one position to the right in a cyclic manner,
        creating a circular permutation pattern.
    """
    p_mat = np.zeros((n, n), dtype=int)
    np.fill_diagonal(p_mat[1:], 1)
    p_mat[0, -1] = 1
    return p_mat
