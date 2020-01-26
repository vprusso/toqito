import numpy as np
from typing import List
from scipy.sparse import identity
from toqito.matrix.operations.vec import vec
from toqito.helper.swap import swap


def apply_map(X: np.ndarray, Phi) -> np.ndarray:
    """
    Applies a superoperator to an operator.
    
    :param X: A matrix.
    :param Phi: A superoperator.
    :return: The operator PHI(X). That is, it is the result of applying the
             superoperator PHI to the operator X.

    PHI should be provided either as a Choi matrix, or as a list of numpy
    arrays with either 1 or 2 columns whose entries are its Kraus operators.
    """

    # Both of the following methods of applying the superoperator are much
    # faster than naively looping through the Kraus operators or constructing
    # eigenvectors of a Choi matrix.

    # The superoperator was given as a list of Kraus operators:
    if isinstance(Phi, list):
        sPhi = [len(Phi), len(Phi[0])]
       
        # Map is completely positive.
        if sPhi[1] == 1 or (sPhi[0] == 1 and sPhi[1] > 2):
            # TODO
            Phi = Phi[:]
            for i in range(sPhi[0]):
                Phi[i][1] = Phi[i][0].conj().T
        else:
            for i in range(sPhi[0]):
                Phi[i][1] = Phi[i][1].conj().T
        Phi_0_list = []
        Phi_1_list = []
        for i in range(sPhi[0]):
            Phi_0_list.append(Phi[i][0])
            Phi_1_list.append(Phi[i][1])

        K_1 = np.concatenate(Phi_0_list, axis=1)
        K_2 = np.concatenate(Phi_1_list, axis=0)

        A = np.kron(np.identity(len(Phi)), X)
        return np.matmul(np.matmul(K_1, A), K_2)

    # The superoperator was given as a Choi matrix:
    if isinstance(Phi, np.ndarray):
        sX = np.array(list(X.shape))
        sNX = np.array(list(Phi.shape)) / sX

        arg_1 = vec(X).T[0]
        arg_2 = np.identity(int(sNX[0]))

        A = np.kron(arg_1, arg_2)
        B = np.reshape(swap(Phi.T,
                            [1, 2],
                            [sX[1], sNX[1], [sX[0], sNX[0]]],
                            True).T,
                    (int(sNX[0]*np.prod(sX)), int(sNX[1])))
        return np.matmul(A, B)

