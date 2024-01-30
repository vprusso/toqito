"""Clock matrix."""
from cmath import exp, pi

import numpy as np


def clock(dim: int) -> np.ndarray:
    r"""Produce clock matrix :cite:`WikiClock`.

    Returns the clock matrix of dimension :code:`dim` described in :cite:`WikiClock`.
    The clock matrix generates the following :code:`dim`-by-:code:`dim` matrix

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
    .. bibliography::
        :filter: docname in docnames


    :param dim: Dimension of the matrix.
    :return: :code:`dim`-by-:code:`dim` clock matrix.

    """
    c_var = 2j * pi / dim
    omega = (exp(k * c_var) for k in range(dim))
    return np.diag(list(omega))
