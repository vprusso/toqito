"""Shift matrix."""
import numpy as np


def shift(dim: int) -> np.ndarray:
    r"""Produce a :code:`dim`-by-:code:`dim` shift matrix :cite:`WikiPauliGen`.

    Returns the shift matrix of dimension :code:`dim` described in :cite:`WikiPauliGen`.
    The shift matrix generates the following :code:`dim`-by-:code:`dim` matrix:

    .. math::
        \Sigma_{1, d} = \begin{pmatrix}
                        0 & 0 & 0 & \ldots & 0 & 1 \\
                        1 & 0 & 0 & \ldots & 0 & 0 \\
                        0 & 1 & 0 & \ldots & 0 & 0 \\
                        0 & 0 & 1 & \ldots & 0 & 0 \\
                        \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
                        0 & 0 & 0 & \ldots & 1 & 0
                    \end{pmatrix}

    The shift matrix is primarily used in the construction of the generalized
    Pauli operators.

    Examples
    ==========

    The shift matrix generated from :math:`d = 3` yields the following matrix:

    .. math::
        \Sigma_{1, 3} =
        \begin{pmatrix}
            0 & 0 & 1 \\
            1 & 0 & 0 \\
            0 & 1 & 0
        \end{pmatrix}

    >>> from toqito.matrices import shift
    >>> shift(3)
    [[0., 0., 1.],
     [1., 0., 0.],
     [0., 1., 0.]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param dim: Dimension of the matrix.
    :return: :code:`dim`-by-:code:`dim` shift matrix.

    """
    shift_mat = np.identity(dim)
    shift_mat = np.roll(shift_mat, -1)
    shift_mat[:, -1] = np.array([0] * dim)
    shift_mat[0, -1] = 1

    return shift_mat
