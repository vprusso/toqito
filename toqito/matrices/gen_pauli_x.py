"""Generalized Pauli-X matrix."""

import numpy as np


def gen_pauli_x(dim: int) -> np.ndarray:
    r"""Produce a :code:`dim`-by-:code:`dim` gen_pauli_x matrix :cite:`WikiPauliGen`.

    Returns the gen_pauli_x matrix of dimension :code:`dim` described in :cite:`WikiPauliGen`.
    The gen_pauli_x matrix generates the following :code:`dim`-by-:code:`dim` matrix:

    .. math::
        \Sigma_{1, d} = \begin{pmatrix}
                        0 & 0 & 0 & \ldots & 0 & 1 \\
                        1 & 0 & 0 & \ldots & 0 & 0 \\
                        0 & 1 & 0 & \ldots & 0 & 0 \\
                        0 & 0 & 1 & \ldots & 0 & 0 \\
                        \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
                        0 & 0 & 0 & \ldots & 1 & 0
                    \end{pmatrix}

    The gen_pauli_x matrix is primarily used in the construction of the generalized
    Pauli operators.

    Examples
    ==========

    The gen_pauli_x matrix generated from :math:`d = 3` yields the following matrix:

    .. math::
        \Sigma_{1, 3} =
        \begin{pmatrix}
            0 & 0 & 1 \\
            1 & 0 & 0 \\
            0 & 1 & 0
        \end{pmatrix}

    >>> from toqito.matrices import gen_pauli_x
    >>> gen_pauli_x(3)
    array([[0., 0., 1.],
           [1., 0., 0.],
           [0., 1., 0.]])

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param dim: Dimension of the matrix.
    :return: :code:`dim`-by-:code:`dim` gen_pauli_x matrix.

    """
    # First column of the identity matrix becomes the last column due to `shift = -1` and `axis=1`
    return np.roll(np.identity(dim), -1, axis=1)
