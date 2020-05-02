"""Produce the CNOT matrix."""
import numpy as np


def cnot() -> np.ndarray:
    r"""
    Produce the CNOT matrix [WikCNOT]_.

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

    >>> from toqito.linear_algebra.matrices.cnot import cnot
    >>> cnot()


    References
    ==========
    .. [WikCNOT] Wikipedia: Controlled NOT gate
        https://en.wikipedia.org/wiki/Controlled_NOT_gate

    :return: The CNOT matrix.
    """
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
