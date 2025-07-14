"""Computes the spark of a matrix."""

from itertools import combinations

import numpy as np


def spark(mat: np.ndarray) -> int:
    """Compute the spark of a matrix.

    The spark of a matrix A is the smallest number of columns from A that are linearly
    dependent :footcite:`Elad_2010_Sparse`.

    Examples
    =========

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import spark

     A = np.array([[1, 0, 1, 2],
                   [0, 1, 1, 3],
                   [1, 1, 2, 5]])

     spark(A)

    Notes
    =====
    - This function only works for 2D NumPy arrays.
    - If all columns are linearly independent, the function returns n_cols + 1.
    - The time complexity of this implementation is O(2^n) in the worst case,
      where n is the number of columns.
    - For an m x n matrix A with n >= m:
      * If spark(A) = m + 1, then rank(A) = m (full rank).
      * spark(A) = 1 if and only if the matrix has a zero column.
      * spark(A) <= rank(A) + 1.

    References
    ==========
    .. footbibliography::


    :param mat: The input matrix as a 2D NumPy array.
    :return: The spark of the input matrix :code:`mat`.
    :raises ValueError: If the input is not a 2D NumPy array.

    """
    if not isinstance(mat, np.ndarray) or mat.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array")

    m, n_cols = mat.shape

    # Check for zero columns
    if np.any(np.all(mat == 0, axis=0)):
        return 1

    for k in range(1, min(m, n_cols) + 1):
        for cols in combinations(range(n_cols), k):
            submatrix = mat[:, cols]
            if np.linalg.matrix_rank(submatrix) < k:
                return k

    # If all columns are linearly independent
    return min(m, n_cols) + 1
