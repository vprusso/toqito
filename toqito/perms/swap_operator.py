import numpy as np
from toqito.helper.iden import iden
from toqito.perms.swap import swap
from typing import List


def swap_operator(dim, is_sparse: bool = False) -> np.ndarray:
    """
    Produces a unitary operator that swaps two subsystems.

    Provides the unitary operator that swaps two copies of DIM-dimensional
    space. If the two subsystems are not of the same dimension, DIM should
    be a 1-by-2 vector containing the dimension of the subsystems.

    :param dim: The dimensions of the subsystems.
    :param is_sparse: Sparse if True and non-sparse if False.
    :return: The swap operator of dimension DIM.
    """

    # Allow the user to enter a single number for dimension.
    if isinstance(dim, int):
        dim = np.array([dim, dim])

    # Swap the rows of Id appropriately.
    return swap(iden(np.prod(dim), is_sparse), [1, 2], dim, True)
