"""Compute the pi-permutation operator."""
import numpy as np
from toqito.matrix.operations.tensor import tensor_n, tensor_list


def pi_perm(dim: int) -> np.ndarray:
    """
    A cycle with only two elements is called a transposition.

    Gives the unitary operator π which is defined by the action
        π(y1 ⊗ y2 ⊗ x1 ⊗ x2) = y1 ⊗ x1 ⊗ y2 ⊗ x2
    for all y1 ∈ Y1, y2 ∈ Y2, x1 ∈ X1, x2 ∈ X2.)

    This permutation operator is used in the semidefinite programs that
    characterize the quantum hedging scenarios described in references [1] and
    [2].

    References:
    [1] Wikipedia page for Cyclic permutations (transpositions):
        https://en.wikipedia.org/wiki/Cyclic_permutation#Transpositions

    [2] Arunachalam, Srinivasan, Abel Molina, and Vincent Russo.
        "Quantum hedging in two-round prover-verifier interactions."
        arXiv preprint arXiv:1310.7954 (2013).
        https://arxiv.org/abs/1310.7954

    [3] Molina, Abel, and John Watrous.
        "Hedging bets with correlated quantum strategies."
        Proceedings of the Royal Society A: Mathematical,
        Physical and Engineering Sciences 468.2145 (2012): 2614-2629.
        https://arxiv.org/abs/1104.1140

    :param dim: The dimension of the permutation operator.
    :return: A permutation operator of dimension `dim`.
    """

    # Permutes the order for two qubits
    swap_matrix = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])
    if dim == 1:
        return np.identity(dim)

    # Simulates a sorting network of depth `dim - 1`. For `dim = 2`, we switch
    # the order from:
    #    (x_1, y_1), (x_2, y_2) -> (x_1, x_2), (y_1, y_2).
    if dim > 1:
        perm_cell = []
        # The element at depth i of the sorting network makes sure that the
        # first and last i + 1 qubits are in the right place, and doesn't
        # modify the qubits already sorted by previous steps.
        for i in range(dim-1):
            s_tensor = tensor_n(swap_matrix, dim-(i+1))
            t_list = tensor_list([np.identity(2**(i+1)),
                                  s_tensor,
                                  np.identity(2**(i+1))])
            perm_cell.append(t_list)

        # We concatenate the steps of the sorting network.
        perm_mat = perm_cell[0]
        for i in range(1, dim-1):
            perm_mat = np.matmul(perm_mat, perm_cell[i])
        return perm_mat
