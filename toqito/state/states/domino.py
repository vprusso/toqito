"""Produces a Domino state."""
import numpy as np
from toqito.base.ket import ket


def domino(idx: int) -> np.ndarray:
    r"""
    Produce a domino state.

    The orthonormal product basis of domino states is given as

    .. math::

        \begin{equation}
            \begin{aligned}
            |\phi_0\rangle = |1\rangle
                            |1 \rangle \qquad
            |\phi_1\rangle = |0 \rangle
                            \left(\frac{|0 \rangle + |1 \rangle}{\sqrt{2}}
                            \right) & \qquad
            |\phi_2\rangle = |0\rangle
                            \left(\frac{|0\rangle - |1\rangle}{\sqrt{2}}\right)
                            \\
            |\phi_3\rangle = |2\rangle
                            \left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}\right)
                            \qquad
            |\phi_4\rangle = |2\rangle
                            \left(\frac{|0\rangle - |1\rangle}{\sqrt{2}}\right)
                            & \qquad
            |\phi_5\rangle = \left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}\right)
                            |0\rangle \\
            |\phi_6\rangle = \left(\frac{|0\rangle - |1\rangle}{\sqrt{2}}\right)
                            |0\rangle \qquad
            |\phi_7\rangle = \left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}\right)
                            |2\rangle & \qquad
            |\phi_8\rangle = \left(\frac{|0\rangle - |1\rangle}{\sqrt{2}}\right)
                            |2\rangle
            \end{aligned}
        \end{equation}

    Returns one of the following nine domino states depending on the value
    of `idx`:

    References:
        [1] Bennett, Charles H., et al.
        Quantum nonlocality without entanglement.
        Phys. Rev. A, 59:1070–1091, Feb 1999.
        https://arxiv.org/abs/quant-ph/9804053

        [2] Bennett, Charles H., et al.
        "Unextendible product bases and bound entanglement."
        Physical Review Letters 82.26 (1999): 5385.

    :param idx: A parameter in [0, 1, 2, 3, 4, 5, 6, 7, 8]
    """
    e_0, e_1, e_2 = ket(3, 0), ket(3, 1), ket(3, 2)
    if idx == 0:
        return np.kron(e_1, e_1)
    if idx == 1:
        return np.kron(e_0, 1 / np.sqrt(2) * (e_0 + e_1))
    if idx == 2:
        return np.kron(e_0, 1 / np.sqrt(2) * (e_0 - e_1))
    if idx == 3:
        return np.kron(e_2, 1 / np.sqrt(2) * (e_1 + e_2))
    if idx == 4:
        return np.kron(e_2, 1 / np.sqrt(2) * (e_1 - e_2))
    if idx == 5:
        return np.kron(1 / np.sqrt(2) * (e_1 + e_2), e_0)
    if idx == 6:
        return np.kron(1 / np.sqrt(2) * (e_1 - e_2), e_0)
    if idx == 7:
        return np.kron(1 / np.sqrt(2) * (e_0 + e_1), e_2)
    if idx == 8:
        return np.kron(1 / np.sqrt(2) * (e_0 - e_1), e_2)
    raise ValueError("Invalid integer value for Domino state.")
