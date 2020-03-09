"""Computes the sign of a permutation."""
import numpy as np
import scipy as sp


def perm_sign(perm: np.ndarray) -> float:
    """
    Compute the "sign" of a permutation.

    The sign (either -1 or 1) of the permutation `perm` is -1**`inv`, where
    `inv` is the number of inversions contained in `perm`.

    References:
    [1] Wikipedia: Parity of a permutation
    https://en.wikipedia.org/wiki/Parity_of_a_permutation

    :param perm: The permutation vector to be checked.
    :return: The value 1 if the permutation is of even length and the value of
             -1 if the permutation is of odd length.
    """
    iden = np.eye(len(perm))
    return sp.linalg.det(iden[:, np.array(perm)-1])
