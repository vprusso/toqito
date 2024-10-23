"""Perfect matchings refers to ways of grouping an even number of objects into pairs."""

import numpy as np


def perfect_matchings(num: list[int] | int | np.ndarray) -> np.ndarray:
    r"""Give all perfect matchings of :code:`num` objects.

    The input can be either an even natural number (the number of objects to be matched) or a `numpy` array containing
    an even number of distinct objects to be matched.

    Returns all perfect matchings of a given list of objects. That is, it returns all ways of grouping an even number of
    objects into pairs.

    This function is adapted from QETLAB. :cite:`QETLAB_link`.

    Examples
    ==========
    This is an example of how to generate all perfect matchings of the numbers 0, 1, 2, 3.

    >>> from toqito.perms import perfect_matchings
    >>> perfect_matchings(4)
    array([[0, 1, 2, 3],
           [0, 2, 1, 3],
           [0, 3, 2, 1]])

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param num: Either an even integer, indicating that you would like all perfect matchings of the
                integers 0, 1, ... N-1, or a `list` or `np.array` containing an even number of distinct
                entries, indicating that you would like all perfect matchings of those entries.
    :return: An array containing all valid perfect matchings of size :code:`num`.

    """
    if isinstance(num, int):
        num = np.arange(num)
    if isinstance(num, list):
        num = np.array(num)

    # Base case, `num = 2`: only one perfect matching.
    if (len_num := len(num)) == 2:
        return num

    # There are no perfect matchings of an odd number of objects.
    if len_num % 2 == 1:
        return np.zeros((0, len_num))

    # Recursive step: build perfect matchings from smaller ones.

    # Only do the recursive step once instead of `num-1` times: we will then tweak
    # the output n-1 times.
    lower_fac = perfect_matchings(num[2:])
    if len(lower_fac.shape) == 1:
        lfac_size = 1
    else:
        lfac_size = lower_fac.shape[0]
    matchings = np.zeros((0, len_num), dtype=int)

    # Now build the perfect matchings we actually want.
    for j in range(1, len_num):
        tlower_fac = lower_fac.copy()
        tlower_fac[tlower_fac == num[j]] = num[1]

        one_vec = np.ones((lfac_size, 2), dtype=int) * [num[0], num[j]]
        if lfac_size == 1:
            one_vec = one_vec[0]

        s_vec = np.hstack((one_vec, tlower_fac))
        matchings = np.vstack((matchings, s_vec))

    return matchings

