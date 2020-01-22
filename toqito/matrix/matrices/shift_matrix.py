import numpy as np


def shift_matrix(dim: int) -> np.ndarray:
    """
    Produces a shift matrix.
    :param dim: Dimension of the matrix.

    Returns the shift matrix of dimension DIM described in [1].

    The shift matrix generates the following DIM x DIM matrix

    \Sigma_1 = \begin{pmatrix}
                    0 & 0 & 0 & \ldots & 0 & 1 \\
                    1 & 0 & 0 & \ldots & 0 & 0 \\
                    0 & 1 & 0 & \ldots & 0 & 0 \\
                    0 & 0 & 1 & \ldots & 0 & 0 \\
                    \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
                    0 & 0 & 0 & \ldots & 1 & 0
               \end{pmatrix}

    References:
    [1] Wikipedia: Generalizations of Pauli matrices
        (https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices#Construction:_The_clock_and_shift_matrices).

    """
    shift_mat = np.identity(dim)
    shift_mat = np.roll(shift_mat, -1)
    shift_mat[:, -1] = np.array([0] * dim)
    shift_mat[0, -1] = 1

    return shift_mat
