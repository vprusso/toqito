"""Calculate permutation sign."""
import numpy as np
from scipy import linalg


def perm_sign(perm: np.ndarray | list[int]) -> float:
    """Compute the "sign" of a permutation :cite:`WikiParPerm`.

    The sign (either -1 or 1) of the permutation :code:`perm` is :code:`-1**`inv`, where :code:`inv` is the number of
    inversions contained in :code:`perm`.

    Examples
    ==========

    For the following vector

    .. math::
        [1, 2, 3, 4]

    the permutation sign is positive as the number of elements in the vector are even. This can be performed in
    :code:`toqito` as follows.

    >>> from toqito.perms import perm_sign
    >>> perm_sign([1, 2, 3, 4])
    1

    For the following vector

    .. math::
        [1, 2, 3, 4, 5]

    the permutation sign is negative as the number of elements in the vector are odd. This can be performed in
    :code:`toqito` as follows.

    >>> from toqito.perms import perm_sign
    >>> perm_sign([1, 2, 4, 3, 5])
    -1

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param perm: The permutation vector to be checked.
    :return: The value 1 if the permutation is of even length and the value of
             -1 if the permutation is of odd length.

    """
    return linalg.det(np.eye(len(perm))[:, np.array(perm) - 1])
