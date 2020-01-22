import numpy as np
from toqito.matrix_ops.tensor import tensor_n, tensor, tensor_list

def pi_perm(n: int) -> np.ndarray:
    """
    Computes the permutation operatr that is used in the hedging SDPs in both
    [1] and [2] to properly align the complex Euclidean spaces.
    References:
        [1]: https://arxiv.org/abs/1310.7954
        [2]: https://arxiv.org/abs/1104.1140
    """

    # Permutes the order for two qubits
    swap_matrix = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])
    if n == 1:
        return np.identity(n)
    # Simultates a sorting network of depth n - 1. For n = 2, we swithc the
    # order from:
    #    (x_1, y_1), (x_2, y_2) -> (x_1, x_2), (y_1, y_2).
    elif n > 1:
        perm_cell = []
        # The element at depth i of the sorting network makes sure that the 
        # first and last i + 1 qubits are in the right place, and doesn't
        # modify the qubits already sorted by previous steps.
        for i in range(n):
            s_tensor = tensor_n(swap_matrix, n-i)
            t_list = [np.identity(2**i), s_tensor, np.identity(2**i)]
            perm_cell.append(tensor_list(t_list))

        # We concatenate the steps of the sorting network.
        perm_mat = perm_cell[0]
        for i in range(1, n):
            perm_mat = perm_mat * perm_cell[i]
        return perm_mat
