"""Produces a Bell state."""
import numpy as np
from toqito.base.ket import ket


def bell(idx: int) -> np.ndarray:
    r"""
    Produce a Bell state.

    Returns one of the following four Bell states depending on the value
    of `idx`:

    .. math::
        \begin{equation}
            \begin{aligned}
                \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) &
                \qquad &
                \frac{1}{\sqrt{2}} \left( |00 \rangle - |11 \rangle \right) \\
                \frac{1}{\sqrt{2}} \left( |01 \rangle + |10 \rangle \right) &
                \qquad &
                \frac{1}{\sqrt{2}} \left( |01 \rangle - |10 \rangle \right)
            \end{aligned}
        \end{equation}

    References:
        [1] Wikipedia: Bell state
        https://en.wikipedia.org/wiki/Bell_state

    :param idx: A parameter in [0, 1, 2, 3]
    """
    e_0, e_1 = ket(2, 0), ket(2, 1)
    if idx == 0:
        return 1 / np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))
    if idx == 1:
        return 1 / np.sqrt(2) * (np.kron(e_0, e_0) - np.kron(e_1, e_1))
    if idx == 2:
        return 1 / np.sqrt(2) * (np.kron(e_0, e_1) + np.kron(e_1, e_0))
    if idx == 3:
        return 1 / np.sqrt(2) * (np.kron(e_0, e_1) - np.kron(e_1, e_0))
    raise ValueError("Invalid integer value for Bell state.")
