"""Produces the reduction map."""
import numpy as np
from scipy.sparse import identity
from toqito.state.states.max_entangled import max_entangled


def reduction_map(dim: int, k: int = 1) -> np.ndarray:
    r"""
    Produces the reduction map.

    :param dim: A positive integer (the dimension of the reduction map).
    :param k:  If this positive integer is provided, the script will instead
               return the Choi matrix of the following linear map:
                .. math::
                    Phi(X) := K * Tr(X)I - X.
    :return: The reduction map.

    If K = 1, this returns the Choi matrix of the reduction map which is a
    positive map on DIM-by-DIM matrices. For a different value of K, this
    yields the Choi matrix of the map defined by:
        R(X) = K * trace(X) * eye(DIM^2) - X. This map is K-positive.
    """
    psi = max_entangled(dim, True, False)
    return k * identity(dim**2) - psi*psi.conj().T
