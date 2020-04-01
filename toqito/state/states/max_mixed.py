"""Produces the maximally mixed state."""
import numpy as np
from scipy import sparse


def max_mixed(dim: int, is_sparse: bool = False) -> [np.ndarray, sparse.dia.dia_matrix]:
    r"""
    Produce the maximally mixed state.

    Produces the maximally mixed state on of `dim` dimensions. The maximally
    mixed state is defined as

    .. math::
        \frac{1}{d} \begin{pmatrix}
                        1 & 0 & \ldots & 0 \\
                        0 & 1 & \ldots & 0 \\
                        \vdots & \vdots & \ddots & \vdots \\
                        0 & 0 & \ldots & 1
                    \end{pmatrix}

    The maximally mixed state is returned as a sparse matrix if
    `is_sparse = True` and is full if `is_sparse = False`.

    References:
        [1] Scott Aaronson: Lecture 6, Thurs Feb 2: Mixed States
        https://www.scottaaronson.com/qclec/6.pdf

    :param dim: Dimension of the entangled state.
    :param is_sparse: `True` if vector is spare and `False` otherwise.
    :return: The maximally mixed state of dimension `dim`.
    """
    if is_sparse:
        return 1 / dim * sparse.eye(dim)
    return 1 / dim * np.eye(dim)
