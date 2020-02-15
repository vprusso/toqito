"""Produces a Werner state."""
import itertools
import numpy as np

from typing import List, Union
from toqito.perms.permutation_operator import permutation_operator
from toqito.perms.swap_operator import swap_operator


def werner_state(dim: int, alpha: Union[float, List[float]]) -> np.ndarray:
    """
    Produce a Werner state.

    Yields a Werner state with parameter `alpha` acting on `(dim *
    dim)`-dimensional space. More specifically, `rho` is the density operator
    defined by (I - `alpha`*S) (normalized to have trace 1), where I is the
    density operator and S is the operator that swaps two copies of
    `dim`-dimensional space (see swap and swap_operator for example).

    If `alpha` is a vector with p!-1 entries, for some integer p > 1, then a
    multipartite Werner state is returned. This multipartite Werner state is
    the normalization of I - `alpha(1)*P(2)` - ... - `alpha(p!-1)*P(p!)`, where
    P(i) is the operator that permutes p subsystems according to the i-th
    permutation when they are written in lexicographical order (for example,
    the lexicographical ordering when p = 3 is:
        [1, 2, 3], [1, 3, 2], [2, 1,3], [2, 3, 1], [3, 1, 2], [3, 2, 1],

    so P(4) in this case equals permutation_operator(dim, [2, 3, 1]).
    """

    # The total number of permutation operators.
    if isinstance(alpha, float):
        n_fac = 2
    else:
        n_fac = len(alpha) + 1

    # Multipartite Werner state.
    if n_fac > 2:
        # Compute the number of parties from `len(alpha)`.
        n_var = n_fac
        # We won't actually go all the way to `n_fac`.
        for i in range(2, n_fac):
            n_var = n_var//i
            if n_var == i + 1:
                break
            if n_var < i:
                msg = """
                    InvalidAlpha: The `alpha` vector must contain p!-1 entries
                    for some integer p > 1.
                """
                raise ValueError(msg)

        # Done error checking and computing the number of parties -- now
        # compute the Werner state.
        perms = list(itertools.permutations(np.arange(n_var)))
        sorted_perms = np.argsort(perms, axis=1) + 1

        for i in range(2, n_fac):
            rho = np.identity(dim**n_var) - alpha[i-1] \
                  * permutation_operator(dim,
                                         sorted_perms[i, :],
                                         False,
                                         True)
        rho = rho / np.trace(rho)
        return rho
    # Bipartite Werner state.
    return (np.identity(dim**2) - alpha
            * swap_operator(dim, True)) / (dim * (dim - alpha))
