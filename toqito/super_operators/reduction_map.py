from toqito.states.max_entangled import max_entangled
from scipy.sparse import identity
import numpy as np


def reduction_map(dim: int, k: int = 1) -> np.ndarray:
    """
    Produces the reduction map.

    :param dim: A positive integer (the dimension of the reduction map).
    :return: The reduction map.

    If K = 1, this returns the Choi matrix of the reduction map which is a 
    positive map on DIM-by-DIM matrices. For a different value of K, this 
    yields the Choi matrix of the map defined by:
        R(X) = K * trace(X) * eye(DIM^2) - X. This map is K-positive.
    """
    psi = max_entangled(dim, 1, 0)
    return k * identity(dim**2) - psi*psi.conj().T
