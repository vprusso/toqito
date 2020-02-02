"""Produces a unitary operator that permutes subsystems."""
from typing import List, Union
import numpy as np
from toqito.perms.permute_systems import permute_systems
from toqito.helper.iden import iden


def permutation_operator(dim: Union[List[int], int],
                         perm: List[int],
                         inv_perm: bool = False,
                         is_sparse: bool = False) -> np.ndarray:
    """
    Produce a unitary operator that permutes subsystems.

    Generates a unitary operator that permutes the order of subsystems
    according to the permutation vector `perm`, where the ith subsystem has
    dimension `dim[i]`.

    If `inv_perm` = True, it implements the inverse permutation of `perm`. The
    permutation operator return is full is `is_sparse` is False and sparse if
    `is_sparse` is True.

    :param dim: The dimensions of the subsystems to be permuted.
    :param perm: A permutation vector.
    :param inv_perm: Boolean dictating if `perm` is inverse or not.
    :param is_sparse: Boolean indicating if return is sparse or not.
    :return: Permutation operator of dimension `dim`.
    """

    # Allow the user to enter a single number for `dim`.
    if isinstance(dim, int):
        dim = dim * np.ones(max(perm))
    if isinstance(dim, list):
        dim = np.array(dim)

    # Swap the rows of the identity matrix appropriately.
    return permute_systems(iden(int(np.prod(dim)), is_sparse),
                           perm,
                           dim,
                           True,
                           inv_perm)
