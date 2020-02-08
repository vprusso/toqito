"""Produces a maximally entangled bipartite pure state."""
import numpy as np
import scipy as sp
from toqito.helper.iden import iden


def max_entangled(dim: int,
                  is_sparse: bool = False,
                  is_normalized: bool = True) -> [np.ndarray, sp.sparse.dia.dia_matrix]:
    """
    Produces a maximally entangled bipartite pure state
    :param dim: Dimension of the entangled state.
    :param is_sparse: `True` if vector is spare and `False` otherwise.
    :param is_normalized: `True` if vector is normalized and `False` otherwise.

    Produces a maximally entangled pure state as above that is sparse
    if `is_sparse = True` and is full is `is_sparse = False`. The pure state
    is normalized to have Euclidean norm 1 if `is_normalized = True`, and it
    is unnormalized (i.e. each entry in the vector is 0 or 1 and the
    Euclidean norm of the vector is `sqrt(dim)` if `is_normalized = False`.
    """
    psi = np.reshape(iden(dim, is_sparse), (dim**2, 1))
    if is_normalized:
        psi = psi/np.sqrt(dim)
    return psi
