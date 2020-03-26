"""Computes a sparse or full identity matrix."""
import numpy as np

from scipy import sparse


def iden(dim: int, is_sparse: bool = False) -> np.ndarray:
    r"""
    Calculate the `dim`-by-`dim` identity matrix.

    Returns the `dim`-by-`dim` identity matrix. If `is_sparse = False` then
    the matrix will be full. If `is_sparse = True` then the matrix will be
    sparse.

    .. math::
        \Sigma_1 = \begin{pmatrix}
                        1 & 0 & 0 & \ldots & 0 \\
                        0 & 1 & 0 & \ldots & 0 \\
                        0 & 0 & 1 & \ldots & 0 \\
                        \vdots & \vdots & \vdots & \ddots & \vdots \\
                        0 & 0 & 0 & \ldots & 1
                   \end{pmatrix}

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
