"""Compute spark of matrix."""

import numpy as np
from itertools import combinations


def spark(A: np.ndarray) -> int:
    """
    Compute the spark of a matrix.

    The spark of a matrix A is the smallest number of columns from A that are linearly dependent [Elad_2010_Sparse].

    Parameters:
    ===========
    A : np.ndarray
        The input matrix as a NumPy array.

    Returns:
    ========
    int
        The spark of the matrix.

    Examples:
    =========
    >>> import numpy as np
    >>> from your_module import spark
    >>> A = np.array([[1, 0, 1, 2],
    ...               [0, 1, 1, 3],
    ...               [1, 1, 2, 5]])
    >>> spark(A)
    3

    Notes:
    -----
    - If all columns are linearly independent, the function returns n_cols + 1.
    - The time complexity of this implementation is O(2^n) in the worst case, 
      where n is the number of columns.

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    """
    n_cols = A.shape[1]
    
    for k in range(1, n_cols + 1):
        for cols in combinations(range(n_cols), k):
            submatrix = A[:, cols]
            if np.linalg.matrix_rank(submatrix) < k:
                return k

    # If all columns are linearly independent
    return n_cols + 1
