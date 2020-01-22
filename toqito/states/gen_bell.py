import numpy as np
from toqito.matrix.matrices.gen_pauli import gen_pauli
from toqito.matrix.operations.vec import vec


def gen_bell(k_1: int, k_2: int, dim: int) -> np.ndarray:
    W = gen_pauli(k_1, k_2, dim)
    return 1/dim * vec(W) * vec(W).conj().T
