"""Computes a sparse or full identity matrix."""
import numpy as np

from scipy import sparse


def iden(dim: int, is_sparse: bool = False) -> np.ndarray:
    """
    Calculate the `dim`-by-`dim` identity matrix.

    Returns the `dim`-by-`dim` identity matrix. If `is_sparse = False` then
    the matrix will be full. If `is_sparse = True` then the matrix will be
    sparse.

    Only use this function within other functions to easily get the correct
    identity matrix. If you always want either the full or the sparse
    identity matrix, just use numpy's built-in np.identity function.

    References:
    [1] Wikipedia: Identity matrix
        https://en.wikipedia.org/wiki/Identity_matrix

    :param dim: Integer representing dimension of identity matrix.
    :param is_sparse: Whether or not the matrix is sparse.
    :return: Sparse identity matrix of dimension `dim`.
    """
    if is_sparse:
        id_mat = sparse.eye(dim)
    else:
        id_mat = np.identity(dim)
    return id_mat
