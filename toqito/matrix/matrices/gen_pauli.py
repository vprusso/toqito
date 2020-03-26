"""Produces a generalized Pauli operator (sometimes called a Weyl operator)."""
import numpy as np

from numpy.linalg import matrix_power
from toqito.matrix.matrices.clock import clock
from toqito.matrix.matrices.shift import shift


def gen_pauli(k_1: int, k_2: int, dim: int) -> np.ndarray:
    """
    Produce generalized Pauli operator.

    Generates a `dim`-by-`dim` unitary operator. More specifically, it is the
    operator X^IND1*Z^IND2, where X and Z are the "shift" and "clock" operators
    that naturally generalize the Pauli X and Z operators. These matrices span
    the entire space of `dim`-by-`dim` matrices as `k_1` and `k_2` range from 0
    to `dim-1`, inclusive.

    References:
        [1] Wikipedia: Generalizations of Pauli matrices
        https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices

    :param k_1: (a non-negative integer from 0 to `dim-1` inclusive).
    :param k_2: (a non-negative integer from 0 to `dim-1` inclusive).
    :param dim: (a positive integer indicating the dimension).
    :return: A generalized Pauli operator.
    """
    gen_pauli_x = shift(dim)
    gen_pauli_z = clock(dim)

    gen_pauli_w = np.matmul(
        matrix_power(gen_pauli_x, k_1), matrix_power(gen_pauli_z, k_2)
    )

    return gen_pauli_w
