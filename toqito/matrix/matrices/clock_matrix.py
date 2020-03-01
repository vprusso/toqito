"""Generates the clock matrix."""
from cmath import exp, pi
import numpy as np


def clock_matrix(dim: int) -> np.ndarray:
    r"""
    Produce clock matrix.

    Returns the clock matrix of dimension `dim` described in [1]. The clock
    matrix generates the following `dim`-by-`dim` matrix

    .. math::
         \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4),
        \Sigma_1 = \begin{pmatrix}
                        1 & 0 & 0 \ldots & 0 \\
                        0 & \omega & \ldots & 0 \\
                        0 & 0 & \omega^2 \ldots & 0 \\
                        \ddots & \ldots & \vdots & \ddots \\
                        0 & 0 & 0 & \ldots & \omega^{d-1}
                   \end{pmatrix}

    where :math: `\omega`  is the n-th primitive root of unity.

    References:
    [1] Wikipedia: Generalizations of Pauli matrices,
        https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices

    :param dim: Dimension of the matrix.
    :return: `dim`-by-`dim` clock matrix.
    """
    c_var = 2j * pi / dim
    omega = (exp(k * c_var) for k in range(dim))
    return np.diag(list(omega))
