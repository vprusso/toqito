"""Computes the negativity of a bipartite density matrix."""
from typing import List, Union
from numpy import linalg as lin_alg
import numpy as np

from toqito.states.pure_to_mixed import pure_to_mixed
from toqito.super_operators.partial_transpose import partial_transpose


def negativity(rho: np.ndarray, dim: Union[List[int], int] = None) -> float:
    """
    Compute the negativity of a bipartite quantum state.

    Calculate the negativity of the quantum state `rho`, assuming that the two
    subsystems on which `rho` acts are of equal dimension (if the local
    dimensions are unequal, specify them in the optional `dim` argument). The
    negativity of `rho` is the sum of the absolute value of the negative
    eigenvalues of the partial transpose of `rho`.

    :param rho: A density matrix of a pure state vector.
    :param dim: The default has both subsystems of equal dimension.
    :return: A value between 0 and 1 that corresponds to the negativity of
            `rho`.
    """
    # Allow the user to input either a pure state vector or a density matrix.
    rho = pure_to_mixed(rho)
    rho_dims = rho.shape
    round_dim = np.round(np.sqrt(rho_dims))

    if dim is None:
        dim = np.array([round_dim])
        dim = dim.T
    if isinstance(dim, list):
        dim = np.array(dim)

    # Allow the user to enter a single number for dim.
    if isinstance(dim, int):
        dim = np.array([dim, rho_dims[0]/dim])
        if abs(dim[1] - np.round(dim[1])) >= 2*rho_dims[0]*np.finfo(float).eps:
            msg = """
                InvalidDim: If `dim` is a scalar, `rho` must be square and `dim`
                must evenly divide `len(rho)`. Please provide the `dim` array
                containing the dimensions of the subsystems.
            """
            raise ValueError(msg)
        dim[1] = np.round(dim[1])

    if np.prod(dim) != rho_dims[0]:
        msg = """
            InvalidDim: Please provide local dimensions in the argument `dim`
            that matach the size of `rho`.
        """
        raise ValueError(msg)

    # Compute the negativity.
    return (lin_alg.norm(partial_transpose(rho, 2, dim), ord="nuc") - 1)/2
