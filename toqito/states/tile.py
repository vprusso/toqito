"""Tile state."""
import numpy as np

from toqito.states import basis


def tile(idx: int) -> np.ndarray:
    r"""
    Produce a Tile state [UPBTile99]_.

    The Tile states constitute five states on 3-by-3 dimensional space that form a UPB
    (unextendible product basis).

    Returns one of the following five tile states depending on the value of :code:`idx`:

    .. math::
        \begin{equation}
            \begin{aligned}
                |\psi_0 \rangle = \frac{1}{\sqrt{2}} |0 \rangle
                \left(|0\rangle - |1\rangle \right),
                \qquad &
                |\psi_1\rangle = \frac{1}{\sqrt{2}}
                \left(|0\rangle - |1\rangle \right) |2\rangle, \\
                |\psi_2\rangle = \frac{1}{\sqrt{2}} |2\rangle
                \left(|1\rangle - |2\rangle \right),
                \qquad &
                |\psi_3\rangle = \frac{1}{\sqrt{2}}
                \left(|1\rangle - |2\rangle \right) |0\rangle, \\
                \qquad &
                |\psi_4\rangle = \frac{1}{3}
                \left(|0\rangle + |1\rangle + |2\rangle)\right)
                \left(|0\rangle + |1\rangle + |2\rangle \right).
            \end{aligned}
        \end{equation}

    Examples
    ==========

    When :code:`idx = 0`, this produces the following tile state

    .. math::
        \frac{1}{\sqrt{2}} |0\rangle \left( |0\rangle - |1\rangle \right).

    Using :code:`toqito`, we can see that this yields the proper state.

    >>> from toqito.states import tile
    >>> import numpy as np
    >>> tile(0)
    [[ 0.70710678]
     [-0.        ]
     [ 0.        ]]

    References
    ==========
    .. [UPBTile99] Bennett, Charles H., et al.
        "Unextendible product bases and bound entanglement."
        Physical Review Letters 82.26 (1999): 5385.
        https://arxiv.org/abs/quant-ph/9808030

    :param idx: A parameter in [0, 1, 2, 3, 4]
    :return: Tile state.
    """
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    if idx == 0:
        return 1 / np.sqrt(2) * np.kron(e_0, (e_0 - e_1))
    if idx == 1:
        return 1 / np.sqrt(2) * np.kron((e_0 - e_1), e_2)
    if idx == 2:
        return 1 / np.sqrt(2) * np.kron(e_2, (e_1 - e_2))
    if idx == 3:
        return 1 / np.sqrt(2) * np.kron((e_1 - e_2), e_0)
    if idx == 4:
        return 1 / 3 * np.kron((e_0 + e_1 + e_2), (e_0 + e_1 + e_2))
    raise ValueError("Invalid integer value for Tile state.")
