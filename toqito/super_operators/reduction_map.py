"""Produces the reduction map."""
import numpy as np
from scipy.sparse import identity
from toqito.state.states.max_entangled import max_entangled


def reduction_map(dim: int, k: int = 1) -> np.ndarray:
    r"""
    Produce the reduction map.

    If `k = 1`, this returns the Choi matrix of the reduction map which is a
    positive map on `dim`-by-`dim` matrices. For a different value of `k`, this
    yields the Choi matrix of the map defined by:

    .. math::
        R(X) = k * \text{Tr}(X) * \mathbb{I} - X,

    where :math:`\mathbb{I}` is the identity matrix. This map is
    :math:`k`-positive.

    Examples
    ==========

    Using `toqito`, we can generate the $3$-dimensional (or standard) reduction
    map as follows.

    >>> from toqito.super_operators.reduction_map import reduction_map
    >>> reduction_map(3).toarray()
    array([[ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.],
           [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
           [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
           [-1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.]])

    :param dim: A positive integer (the dimension of the reduction map).
    :param k:  If this positive integer is provided, the script will instead
               return the Choi matrix of the following linear map:
               Phi(X) := K * Tr(X)I - X.
    :return: The reduction map.
    """
    psi = max_entangled(dim, True, False)
    return k * identity(dim ** 2) - psi * psi.conj().T
