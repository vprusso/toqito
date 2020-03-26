"""Generates the shift matrix."""
import numpy as np


def shift(dim: int) -> np.ndarray:
    r"""
    Produce a `dim`-by-`dim` shift matrix.

    Returns the shift matrix of dimension `dim` described in [1]. The shift
    matrix generates the following `dim`-by-`dim` matrix:

    .. math::
        \Sigma_1 = \begin{pmatrix}
                        0 & 0 & 0 & \ldots & 0 & 1 \\
                        1 & 0 & 0 & \ldots & 0 & 0 \\
                        0 & 1 & 0 & \ldots & 0 & 0 \\
                        0 & 0 & 1 & \ldots & 0 & 0 \\
                        \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
                        0 & 0 & 0 & \ldots & 1 & 0
                    \end{pmatrix}

    The shift matrix is primarily used in the construction of the generalized
    Pauli operators.

    References:
        [1] Wikipedia: Generalizations of Pauli matrices
        (https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices.

    :param dim: Dimension of the matrix.
    :return: `dim`-by-`dim` shift matrix.
    """
    shift_mat = np.identity(dim)
    shift_mat = np.roll(shift_mat, -1)
    shift_mat[:, -1] = np.array([0] * dim)
    shift_mat[0, -1] = 1

    return shift_mat
