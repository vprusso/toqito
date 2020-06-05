"""Calculate permutation sign."""
from typing import List, Union
from scipy import linalg

import numpy as np


def perm_sign(perm: Union[np.ndarray, List[int]]) -> float:
    """
    Compute the "sign" of a permutation [WikParPerm]_.

    The sign (either -1 or 1) of the permutation :code:`perm` is -1**`inv`, where `inv` is the
    number of inversions contained in :code:`perm`.

    Examples
    ==========

    For the following vector

    .. math::
        [1, 2, 3, 4]

    the permutation sign is positive as the number of elements in the vector are even. This can be
    performed in :code:`toqito` as follows.

    >>> from toqito.perms import perm_sign
    >>> perm_sign([1, 2, 3, 4])
    1

    For the following vector

    .. math::
        [1, 2, 3, 4, 5]

    the permutation sign is negative as the number of elements in the vector are odd. This can be
    performed in :code:`toqito` as follows.

    >>> from toqito.perms import perm_sign
    >>> perm_sign([1, 2, 4, 3, 5])
    -1

    References
    ==========
    .. [WikParPerm] Wikipedia: Parity of a permutation
        https://en.wikipedia.org/wiki/Parity_of_a_permutation

    :param perm: The permutation vector to be checked.
    :return: The value 1 if the permutation is of even length and the value of
             -1 if the permutation is of odd length.
    """
    eye = np.eye(len(perm))
    return linalg.det(eye[:, np.array(perm) - 1])
