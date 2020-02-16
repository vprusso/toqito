from typing import Union
import numpy as np


def perfect_matchings(items: Union[int, np.ndarray]) -> np.ndarray:
    """
    """
    if isinstance(items, int):
        items = np.array(list(range(items)))
    sz = len(items)

    # Base case, number of items is 2. Only on perfect matching.
    if sz == 2:
        return items

    # There are no perfect matchings of an odd number of objects.
    if sz % 2 == 1:
        return np.zeros((0, sz))

    # Recursive step: build perfect matchings from smaller ones.

    # Only do the recursive step once instead of `items-1` times: we will then
    # tweak the output `items-1` times.
    lower_fac = perfect_matchings(items[2:])
    lfac_size = lower_fac.shape[0] - 1
    pm = np.zeros((0, sz))

    # Now build the perfect matchings we actually want.
    for i in range(1, sz):
        tlower_fac = lower_fac
        tlower_fac[tlower_fac == items[i]] = items[1]
        a = pm
        b = np.ones((lfac_size, 1))
        c = np.array([items[0], items[i]])
        d = tlower_fac
        print("PERM", a)
        
        pm = [pm, b * c, d]

    return pm
