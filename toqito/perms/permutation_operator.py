"""Produces a unitary operator that permutes subsystems."""
import numpy as np
from toqito.perms.permute_systems import permute_systems
from toqito.helper.iden import iden
from typing import Any, List


def permutation_operator(dim: Any,
                         perm: List[int],
                         inv_perm: bool = False,
                         is_sparse: bool = False) -> np.ndarray:
    """
    Produce a unitary operator that permutes subsystems.

    Generates a unitary operator that permutes the order of subsystems
    according to the permutation vector PERM, where the ith subsystem has
    dimension DIM(i).

    If INV_PERM = True, it implements the inverse permutation of PERM. The
    permutation operator return is full is IS_SPARSE is False and sparse if
    IS_SPARSE is True.

    :param dim: The dimensions of the subsystems to be permuted.
    :param perm: A permutation vector.
    :param inv_perm: Boolean dictating if PERM is inverse or not.
    :param is_sparse: Boolean indicating if return is sparse or not.
    :return: Permutation operator of dimension DIM.
    """

    # Allow the user to enter a single number for DIM:
    if isinstance(dim, int):
        dim = dim * np.ones(max(perm))

    # Swap the rows of Id appropriately.
    return permute_systems(
            iden(int(np.prod(dim)), is_sparse),
            perm,
            dim,
            True,
            inv_perm)

