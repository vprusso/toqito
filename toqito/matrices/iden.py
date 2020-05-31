"""Identity matrix."""
from scipy import sparse

import numpy as np


def iden(dim: int, is_sparse: bool = False) -> np.ndarray:
    r"""
    Calculate the `dim`-by-`dim` identity matrix [WIKID]_.

    Returns the `dim`-by-`dim` identity matrix. If `is_sparse = False` then
    the matrix will be full. If `is_sparse = True` then the matrix will be
    sparse.

    .. math::
        \mathbb{I} = \begin{pmatrix}
                        1 & 0 & 0 & \ldots & 0 \\
                        0 & 1 & 0 & \ldots & 0 \\
                        0 & 0 & 1 & \ldots & 0 \\
                        \vdots & \vdots & \vdots & \ddots & \vdots \\
                        0 & 0 & 0 & \ldots & 1
                   \end{pmatrix}

    Only use this function within other functions to easily get the correct
    identity matrix. If you always want either the full or the sparse
    identity matrix, just use numpy's built-in np.identity function.

    Examples
    ==========

    The identity matrix generated from :math:`d = 3` yields the following
    matrix:

    .. math::
        \mathbb{I}_3 = \begin{pmatrix}
            1 & 0 & 0 \\
            0 & 1 & 0 \\
            0 & 0 & 1
        \end{pmatrix}

    >>> from toqito.matrices import iden
    >>> iden(3)
    [[1., 0., 0.],
     [0., 1., 0.],
     [0., 0., 1.]])

    It is also possible to create sparse identity matrices. The sparse identity
    matrix generated from :math:`d = 10` yields the following matrix:

    >>> from toqito.matrices import iden
    >>> iden(10, True)
    <10x10 sparse matrix of type '<class 'numpy.float64'>' with 10 stored
    elements (1 diagonals) in DIAgonal format>

    References
    ==========
    .. [WIKID] Wikipedia: Identity matrix
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
