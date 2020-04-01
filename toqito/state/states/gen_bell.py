"""Produces a generalized Bell state."""
import numpy as np
from toqito.matrix.matrices.gen_pauli import gen_pauli
from toqito.matrix.operations.vec import vec


def gen_bell(k_1: int, k_2: int, dim: int) -> np.ndarray:
    """
    Produce a generalized Bell state.

    Produces a generalized Bell state. Note that the standard Bell states
    can be recovered as:

    bell(0) -> gen_bell(0, 0, 2)
    bell(1) -> gen_bell(0, 1, 2)
    bell(2) -> gen_bell(1, 0, 2)
    bell(3) -> gen_bell(1, 1, 2)

    References:
        [1] Sych, Denis, and Gerd Leuchs.
        "A complete basis of generalized Bell states."
        New Journal of Physics 11.1 (2009): 013006.

    :param k_1: An integer 0 <= k_1 <= n.
    :param k_2: An integer 0 <= k_2 <= n.
    :param dim: The dimension of the generalized Bell state.
    """
    gen_pauli_w = gen_pauli(k_1, k_2, dim)
    return 1 / dim * vec(gen_pauli_w) * vec(gen_pauli_w).conj().T
