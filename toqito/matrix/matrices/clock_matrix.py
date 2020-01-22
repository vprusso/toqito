import numpy as np
from cmath import exp, pi


def clock_matrix(dim: int) -> np.ndarray:
    """
    Produces a clock matrix.
    :param dim: Dimension of the matrix.

    Returns the clock matrix of dimension DIM described in [1].

    The clock matrix generates the following DIM x DIM matrix

    \Sigma_1 = \begin{pmatrix}
                 1 & 0 & 0 & \ldots & 0 \\
                 0 & \omega & 0 & \ldots & 0 \\
                 0 & 0 & \omega^2 & \ldots & 0 \\
                 \vdots & \vdots & \vdots & \ddots & \vdots \\ 
                 0 & 0 & 0 & \ldots & \omega^{d-1}
               \end{pmatrix}

    where $\omega$ is the n-th primitive root of unity.

    References:
    [1] Wikipedia: Generalizations of Pauli matrices
        (https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices#Construction:_The_clock_and_shift_matrices).

    """
    c = 2j * pi / dim
    omega = (exp(k * c) for k in range(dim))
    return np.diag(list(omega))

