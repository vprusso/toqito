"""Computes comparison matrix."""

import numpy as np


def comparison(mat: np.ndarray) -> np.ndarray:
    r"""Compute comparison matrix of a given square matrix.

    This function computes the comparison matrix :math:`M(A)` for a square matrix :math:`A` as defined in
    :footcite:`WikiComparisonMatrix`. For each entry, the diagonal entries are given by the absolute value of the
    original diagonal entries of :math:`A`, while the off-diagonal entries are given by minus the absolute value of the
    corresponding entries. In other words,

    .. math::
        m_{ij} =
        \begin{cases}
        |a_{ij}|, & \text{if } i = j, \\
        -|a_{ij}|, & \text{if } i \neq j.
        \end{cases}

    Examples
    ========

    .. jupyter-execute::

        import numpy as np
        from toqito.matrices import comparison
        A = np.array([[2, -1],
                    [3, 4]])
        print(comparison(A))

    References
    ==========
    .. footbibliography::



    :param mat: The input square matrix.
    :raises ValueError: If the input matrix is not square.
    :return: The comparison matrix of the input matrix.

    """
    if len(mat.shape) != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Input matrix must be square.")

    # Compute the matrix whose off-diagonal entries are -|a_{ij}|.
    cmp_mat = -np.abs(mat).astype(float)
    # Replace the diagonal with the absolute values of the original diagonal.
    np.fill_diagonal(cmp_mat, np.abs(np.diag(mat)))
    return cmp_mat
