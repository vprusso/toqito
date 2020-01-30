import numpy as np
import operator
import functools
from toqito.perms.swap import swap
from toqito.super_operators.partial_transpose import partial_transpose


def realignment(X, dim=None):
    """
    """
    eps = np.finfo(float).eps
    dX = X.shape
    round_dim = np.round(np.sqrt(dX))
    if dim is None:
        dim = np.transpose(np.array([round_dim]))
    if isinstance(dim, list):
        dim = np.array(dim)

    if len(dim) == 1:
        dim = np.array([[dim], [dX[0]/dim]])
        if np.abs(dim[1] - np.round(dim[1])) >= 2*dX[0]*eps:
            raise ValueError("InvalidDim:")
        dim[1] = np.round(dim[1])

    if min(dim.shape) == 1:
        dim = dim[:].T
        dim = functools.reduce(operator.iconcat, dim, [])
        dim = np.array([[dim], [dim]])
        dim = functools.reduce(operator.iconcat, dim, [])

    x = np.array([[dim[0][1], dim[0][0]], [dim[1][0], dim[1][1]]])
    y = np.array([[dim[1][0], dim[0][0]], [dim[0][1], dim[1][1]]])
    Z = swap(X, [1, 2], dim, 1)

    return swap(partial_transpose(Z, sys=1, dim=x), [1, 2], y, 1)
