import numpy as np
from toqito.helper.iden import iden


def max_entangled(dim: int,
                  is_sparse: bool = False,
                  is_normalized: bool = True) -> np.ndarray:
    """
    Produces a maximally entangled bipartite pure state
    :param dim: Dimension of the entangled state.
    :param is_sparse: True if vector is spare and False otherwise.
    :param is_normalized: True if vecotr is normalized and False otherwise.

    Produces a maximally entangled pure state as above that is sparse
    if IS_SPARSE = TRUE and is full is IS_SPARSE = FALSE. The pure state
    is normalized to have Euclidean norm 1 if IS_NORMALIZED = TRUE, and it
    is unnormalized (i.e. each entry in the vector is 0 or 1 and the 
    Euclidean norm of the vector is sqrt(DIM)) if IS_NORMALIZED = FALSE.
    """
    psi = np.reshape(iden(dim, is_sparse), (dim**2, 1))
    if is_normalized:
        psi = psi/np.sqrt(dim)
    return psi
