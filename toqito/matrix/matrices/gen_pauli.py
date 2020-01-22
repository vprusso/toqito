import numpy as np
from numpy.linalg import matrix_power
from toqito.matrix.matrices.clock_matrix import clock_matrix
from toqito.matrix.matrices.shift_matrix import shift_matrix


def gen_pauli(k_1: int, k_2: int, dim: int) -> np.ndarray:

    X = shift_matrix(dim)
    Z = clock_matrix(dim)

    W = np.matmul(matrix_power(X, k_1), matrix_power(Z, k_2))

    return W
