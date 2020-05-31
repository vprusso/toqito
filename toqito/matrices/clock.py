"""Clock matrix."""
from cmath import exp, pi

import numpy as np


def clock(dim: int) -> np.ndarray:
    r"""
    Produce clock matrix [WikClock]_.

    Returns the clock matrix of dimension `dim` described in [WikClock]_. The
    clock matrix generates the following `dim`-by-`dim` matrix

    .. math::
        \Sigma_{1, d} = \begin{pmatrix}
                        1 & 0 & 0 & \ldots & 0 \\
                        0 & \omega & 0 & \ldots & 0 \\
                        0 & 0 & \omega^2 & \ldots & 0 \\
                        \vdots & \vdots & \vdots & \ddots & \vdots \\
                        0 & 0 & 0 & \ldots & \omega^{d-1}
                   \end{pmatrix}

    where :math:`\omega` is the n-th primitive root of unity.

    The clock matrix is primarily used in the construction of the generalized
    Pauli operators.

    Examples
    ==========

    The clock matrix generated from :math:`d = 3` yields the following matrix:

    .. math::
        \Sigma_{1, 3} = \begin{pmatrix}
            1 & 0 & 0 \\
            0 & \omega & 0 \\
            0 & 0 & \omega^2
        \end{pmatrix}

    >>> from toqito.matrices import clock
    >>> clock(3)
    [[ 1. +0.j       ,  0. +0.j       ,  0. +0.j       ],
     [ 0. +0.j       , -0.5+0.8660254j,  0. +0.j       ],
     [ 0. +0.j       ,  0. +0.j       , -0.5-0.8660254j]]

    References
    ==========
    .. [WikClock] Wikipedia: Generalizations of Pauli matrices,
        https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices

    :param dim: Dimension of the matrix.
    :return: `dim`-by-`dim` clock matrix.
    """
    c_var = 2j * pi / dim
    omega = (exp(k * c_var) for k in range(dim))
    return np.diag(list(omega))
