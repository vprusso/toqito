"""Produces a Bell state."""
import numpy as np
from toqito.base.ket import ket


def bell(idx: int) -> np.ndarray:
    """
    Produce a Bell state.

    Returns one of the following four Bell states depending on the value
    of `idx`:
        0: (|0>|0> + |1>|1>)/sqrt(2)
        1: (|0>|0> - |1>|1>)/sqrt(2)
        2: (|0>|1> + |1>|0>)/sqrt(2)
        3: (|0>|1> - |1>|0>)/sqrt(2)

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
