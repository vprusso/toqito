"""Checks if matrix forms an equilangular tight frame (ETF)."""
from toqito.matrix_ops import vectors_to_gram_matrix
import numpy as np


def is_etf(mat: np.ndarray, rtol: float = 1e-9, atol: float = 1e-9) -> bool:
    """Check if a matrix constitutes an equilangular tight frame (ETF).

    Definition taken from:
    :cite:`hoffman2011equilangular`.

    A matrix :math:`A` constitutes an equilangular tight frame if it satisfies three conditions:

    1. Each column of the matrix :math:`A` has unit norm.
    2. All diagonal elements of the Gram matrix are one, and its off-diagonal elements are constant.
    3. :math:`AA^* = (ncols/nrows) I`. Here, :math:`A^*` is the conjugate transpose of :math:`A`
       and :math:`I` is the identity matrix.

    Example
    ========
    >>> # Example of a ETF matrix.
    >>> import numpy as np
    >>> from toqito.matrix_ops import vectors_to_gram_matrix
    >>> from toqito.matrix_props import is_etf
    >>> mat = np.array([[1, -1/2, -1/2], [0, np.sqrt(3)/2, -np.sqrt(3)/2]])
    >>> is_etf(mat)
    True

    References
    ==========
    .. bibliography::
        :filter: docname in docnames
        

    Args:
        mat (np.ndarray): The matrix to check.
        rtol: Relative tolerance for numerical comparisons (default: 1e-9).
        atol: Absolute tolerance for numerical comparisons (default: 1e-9).
        
    Returns:
        bool: True if the matrix constitutes an ETF, False otherwise.
        
    """
    nrows, ncols = mat.shape

    # Check if each column has unit norm.
    for col in range(ncols):
        if not np.isclose(np.linalg.norm(mat[:, col]), 1, rtol=rtol, atol=atol):
            return False

    col_vectors = [mat[:, i] for i in range(mat.shape[1])]
    gram_matrix = vectors_to_gram_matrix(col_vectors)
    diag_gram_matrix = np.diag(gram_matrix)

    # Check if diagonal elements of gram matrix are ones.
    if not np.allclose(diag_gram_matrix, 1, rtol=rtol, atol=atol):
        return False

    # Check if off-diagonal elements are constant in magnitude.
    off_diag_elements = gram_matrix[~np.eye(gram_matrix.shape[0], dtype=bool)]
    mod_off_diag = np.abs(off_diag_elements)

    if not np.allclose(mod_off_diag, mod_off_diag[0], rtol=rtol, atol=atol):
        return False

    # Verify tight frame condition AA* = (ncols/nrows) * Identity
    return np.allclose(mat @ mat.conj().T, (ncols / nrows) * np.identity(nrows), rtol=rtol, atol=atol)
