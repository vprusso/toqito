"""Produces a generalized Pauli-Z operator matrix."""

from cmath import exp, pi

import numpy as np


def gen_pauli_z(dim: int) -> np.ndarray:
    r"""Produce gen_pauli_z matrix :footcite:`WikiClock`.

    Returns the gen_pauli_z matrix of dimension :code:`dim` described in :footcite:`WikiClock`.
    The gen_pauli_z matrix generates the following :code:`dim`-by-:code:`dim` matrix

    .. math::
        \Sigma_{1, d} = \begin{pmatrix}
                        1 & 0 & 0 & \ldots & 0 \\
                        0 & \omega & 0 & \ldots & 0 \\
                        0 & 0 & \omega^2 & \ldots & 0 \\
                        \vdots & \vdots & \vdots & \ddots & \vdots \\
                        0 & 0 & 0 & \ldots & \omega^{d-1}
                   \end{pmatrix}

    where :math:`\omega` is the n-th primitive root of unity.

    The gen_pauli_z matrix is primarily used in the construction of the generalized
    Pauli operators.

    Examples
    ==========

    The gen_pauli_z matrix generated from :math:`d = 3` yields the following matrix:

    .. math::
        \Sigma_{1, 3} = \begin{pmatrix}
            1 & 0 & 0 \\
            0 & \omega & 0 \\
            0 & 0 & \omega^2
        \end{pmatrix}

    .. jupyter-execute::

     from toqito.matrices import gen_pauli_z

     gen_pauli_z(3)

    References
    ==========
    .. footbibliography::



    :param dim: Dimension of the matrix.
    :return: :code:`dim`-by-:code:`dim` gen_pauli_z matrix.

    """
    c_var = 2j * pi / dim
    omega = (exp(k * c_var) for k in range(dim))
    return np.diag(list(omega))
