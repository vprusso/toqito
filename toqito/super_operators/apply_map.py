"""Applies a superoperator to an operator."""
from typing import List, Union
import numpy as np
from toqito.matrix.operations.vec import vec
from toqito.perms.swap import swap


def apply_map(mat: np.ndarray,
              phi_op: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    """
    Applies a superoperator to an operator.

    :param mat: A matrix.
    :param phi_op: A superoperator.
    :return: The result of applying the superoperator `phi_op` to the operator
             `mat`.

    `phi_op` should be provided either as a Choi matrix, or as a list of numpy
    arrays with either 1 or 2 columns whose entries are its Kraus operators.
    """

    # Both of the following methods of applying the superoperator are much
    # faster than naively looping through the Kraus operators or constructing
    # eigenvectors of a Choi matrix.

    # The superoperator was given as a list of Kraus operators:
    if isinstance(phi_op, list):
        s_phi_op = [len(phi_op), len(phi_op[0])]

        # Map is completely positive.
        if s_phi_op[1] == 1 or (s_phi_op[0] == 1 and s_phi_op[1] > 2):
            for i in range(s_phi_op[0]):
                phi_op[i][1] = phi_op[i][0].conj().T
        else:
            for i in range(s_phi_op[0]):
                phi_op[i][1] = phi_op[i][1].conj().T
        Phi_0_list = []
        Phi_1_list = []
        for i in range(s_phi_op[0]):
            Phi_0_list.append(phi_op[i][0])
            Phi_1_list.append(phi_op[i][1])

        K_1 = np.concatenate(Phi_0_list, axis=1)
        K_2 = np.concatenate(Phi_1_list, axis=0)

        A = np.kron(np.identity(len(phi_op)), mat)
        return np.matmul(np.matmul(K_1, A), K_2)

    # The superoperator was given as a Choi matrix:
    if isinstance(phi_op, np.ndarray):
        sX = np.array(list(mat.shape))
        sNX = np.array(list(phi_op.shape)) / sX

        arg_1 = vec(mat).T[0]
        arg_2 = np.identity(int(sNX[0]))

        A = np.kron(arg_1, arg_2)
        sys = [1, 2]
        dim = [[sX[1], sNX[1]], [sX[0], sNX[0]]]
        B = np.reshape(swap(phi_op.T,
                            sys,
                            dim,
                            True).T,
                       (int(sNX[0]*np.prod(sX)), int(sNX[1])))
        return np.matmul(A, B)
