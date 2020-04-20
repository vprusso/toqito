"""Computes the sign of a permutation."""
from typing import List, Union
import numpy as np
from scipy import linalg


def perm_sign(perm: Union[np.ndarray, List[int]]) -> float:
    """
    Compute the "sign" of a permutation [WIKPARP]_.

    The sign (either -1 or 1) of the permutation `perm` is -1**`inv`, where
    `inv` is the number of inversions contained in `perm`.

    Examples
    ==========

    For the following vector

    .. math::
        [1, 2, 3, 4]

    the permutation sign is positive as the number of elements in the vector are
    even. This can be performed in `toqito` as follows.

    >>> from toqito.perms.perm_sign import perm_sign
    >>> perm_sign([1, 2, 3, 4])
    1

    For the following vector

    .. math::
        [1, 2, 3, 4, 5]

    the permutation sign is negative as the number of elements in the vector are
    odd. This can be performed in `toqito` as follows.

    >>> from toqito.perms.perm_sign import perm_sign
    >>> perm_sign([1, 2, 4, 3, 5])
    -1

    References
    ==========
    .. [WIKPARP] Wikipedia: Parity of a permutation
        https://en.wikipedia.org/wiki/Parity_of_a_permutation

    :param perm: The permutation vector to be checked.
    :return: The value 1 if the permutation is of even length and the value of
             -1 if the permutation is of odd length.
    """
    iden = np.eye(len(perm))
    return linalg.det(iden[:, np.array(perm) - 1])
