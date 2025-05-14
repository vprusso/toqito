"""CNOT matrix generates the CNOT operator matrix."""

import numpy as np


def cnot() -> np.ndarray:
    r"""Produce the CNOT matrix :cite:`WikiCNOT`.

    The CNOT matrix is defined as

    .. math::
        \text{CNOT} =
        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1 \\
            0 & 0 & 1 & 0
        \end{pmatrix}.

    Examples
    ==========
    .. jupyter-execute::

     from toqito.matrices import cnot

     cnot()


    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :return: The CNOT matrix.

    """
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
