'''Checks if matrix forms an equilangular tight frame (ETF).'''
import numpy as np
from toqito.matrix_ops import vectors_to_gram_matrix

def is_etf(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check if matrix constitutes an equilangular tight frame.
    
    Args:
        mat(array): A matrix of any size
    
    Definition taken from:
    :cite:`http://users.cms.caltech.edu/~jtropp/conf/Tro05-Complex-Equiangular-SPIE-preprint.pdf`.
    
    A matrix A of any size constitutes an equilangular tight frame if it satisfies three conditions:
        1) If each of the columns of the matrix has unit norm. 
        2) If all the diagnol elements of the gram matrix is one and its off-diagnol elements are constant.
        3) If A.A^* = (ncols / nrows). I(Identity matrix)
    
    where:
        - A^* is the conjugate transpose (Hermitian transpose) of A.
    
    Examples
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
        
    :param mat: The matrix to check.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: Return :code:True if the matrix constitutes ETF, and :code:False otherwise.
    
    """
    nrows, ncols = mat.shape[0], mat.shape[1]
    
    #Checks if the each column has a unit norm.
    for col in range(ncols):
        if not np.isclose(np.linalg.norm(mat[:,col]), 1):
            return False

        
    #Checks if the matrix is equiangular
    col_vectors = [mat[:, i] for i in range(mat.shape[1])]
    if not np.allclose(np.diag(vectors_to_gram_matrix(col_vectors)), 1):
        return False
    
    off_diagonal_elements = vectors_to_gram_matrix(col_vectors)[~np.eye(vectors_to_gram_matrix(col_vectors).shape[0], dtype=bool)]
    mod_off_diagonal = np.abs(off_diagonal_elements)
    if not np.allclose(mod_off_diagonal, mod_off_diagonal[0]):
        return False
    
    return np.allclose(mat @ mat.conj().T, (ncols / nrows) * np.identity(nrows))
  