"""Produces a unitary operator that swaps two subsystems."""
from typing import List, Union
import numpy as np
from toqito.matrix.matrices.iden import iden
from toqito.perms.swap import swap


def swap_operator(dim: Union[List[int], int], is_sparse: bool = False) -> np.ndarray:
    """
    Produce a unitary operator that swaps two subsystems.

    Provides the unitary operator that swaps two copies of `dim`-dimensional
    space. If the two subsystems are not of the same dimension, `dim` should
    be a 1-by-2 vector containing the dimension of the subsystems.

    :param dim: The dimensions of the subsystems.
    :param is_sparse: Sparse if `True` and non-sparse if `False`.
    :return: The swap operator of dimension `dim`.
    """
    # Allow the user to enter a single number for dimension.
    if isinstance(dim, int):
        dim = np.array([dim, dim])

    # Swap the rows of the identity appropriately.
    return swap(iden(int(np.prod(dim)), is_sparse), [1, 2], dim, True)
