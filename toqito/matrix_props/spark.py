"""Compute spark of matrix."""

from itertools import combinations

import numpy as np


def spark(mat: np.ndarray) -> int:
    """Compute the spark of a matrix.

    The spark of a matrix A is the smallest number of columns from A that are linearly dependent [Elad_2010_Sparse].

    Examples
    =========
    >>> import numpy as np
    >>> from toqito.matrix_props import spark
    >>> A = np.array([[1, 0, 1, 2],
    ...               [0, 1, 1, 3],
    ...               [1, 1, 2, 5]])
    >>> spark(A)
    3

    Notes
    =====
    - If all columns are linearly independent, the function returns n_cols + 1.
    - The time complexity of this implementation is O(2^n) in the worst case,
      where n is the number of columns.

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param mat: The input matrix.
    :return: The spark of the input matrix :code:`mat`.

    """
    n_cols = mat.shape[1]

    for k in range(1, n_cols + 1):
        for cols in combinations(range(n_cols), k):
            submatrix = mat[:, cols]
            if np.linalg.matrix_rank(submatrix) < k:
                return k

    # If all columns are linearly independent
    return n_cols + 1
