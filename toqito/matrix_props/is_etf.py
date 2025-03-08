"""Checks if matrix forms an equilangular tight frame (ETF)."""
import numpy as np
from toqito.matrix_ops import vectors_to_gram_matrix


def is_etf(mat: np.ndarray) -> bool:
    r"""
    Checks if a matrix constitutes an equilangular tight frame(ETF).
    
    Definition taken from:
    :cite:`hoffman2011complexequiangulartightframes`.
    
    A matrix math::`A` constitutes an equilangular tight frame if it satisfies three conditions:
        1) If each of the columns of the matrix math::`A` has unit norm. 
        2) If all the diagonol elements of the gram matrix is one and its off-diagonol elements are constant.
        3) :math:`AA* = (ncols/nrows)I`. Here :math:`A*` is conjugate transpose of :math:`A` and I is Identity matrix.
    
    Examples
    ========
    >>> # Example of an ETF matrix.
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
        
    :param mat: The matrix to check.
    :return: Return True if the matrix constitutes ETF, and False otherwise.
    
    """
    nrows, ncols = mat.shape[0], mat.shape[1]
    
    # Checks if the each column has a unit norm.
    for col in range(ncols):
        if not np.isclose(np.linalg.norm(mat[:,col]), 1):
            return False

        
    # Checks if the matrix is equiangular
    col_vectors = [mat[:, i] for i in range(mat.shape[1])]
    if not np.allclose(np.diag(vectors_to_gram_matrix(col_vectors)), 1):
        return False
    
    off_diagonal_elements = vectors_to_gram_matrix(col_vectors)[~np.eye(vectors_to_gram_matrix(col_vectors).shape[0], dtype=bool)]
    mod_off_diagonal = np.abs(off_diagonal_elements)
    if not np.allclose(mod_off_diagonal, mod_off_diagonal[0]):
        return False
    
    return np.allclose(mat @ mat.conj().T, (ncols / nrows) * np.identity(nrows))
  
