"""Gives all perfect matchings of N objects."""
from typing import Union
import numpy as np


def perfect_matchings(items: Union[int, np.ndarray]) -> np.ndarray:
    """Compute perfect matchings."""
    if isinstance(items, int):
        items = np.array(list(range(items)))

    # Base case, number of items is 2. Only on perfect matching.
    if len(items) == 2:
        return items

    # There are no perfect matchings of an odd number of objects.
    if len(items) % 2 == 1:
        return np.zeros((0, len(items)))

    # Recursive step: build perfect matchings from smaller ones.

    # Only do the recursive step once instead of `items-1` times: we will then
    # tweak the output `items-1` times.
    lower_fac = perfect_matchings(items[2:])
    lfac_size = lower_fac.shape[0] - 1
    p_matches = np.zeros((0, len(items)))

    # Now build the perfect matchings we actually want.
    for i in range(1, len(items)):
        tlower_fac = lower_fac
        tlower_fac[tlower_fac == items[i]] = items[1]
        a_tmp = p_matches
        b_tmp = np.ones((lfac_size, 1))
        c_tmp = np.array([items[0], items[i]])
        d_tmp = tlower_fac
        print("PERM", a_tmp)

        p_matches = [p_matches, b_tmp * c_tmp, d_tmp]

    return p_matches
