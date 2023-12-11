"""Antisymmetric projection operator."""
from itertools import permutations

import numpy as np
from scipy import linalg, sparse

from toqito.perms import perm_sign, permutation_operator


def antisymmetric_projection(dim: int, p_param: int = 2, partial: bool = False) -> sparse.lil_matrix:
    
    dimp = dim ** p_param

    if p_param == 1:
        return sparse.eye(dim)
    # The antisymmetric subspace is empty if `dim < p`.
    if dim < p_param:
        return sparse.lil_matrix((dimp, dimp * (1 - partial)))

    p_list = np.array(list(permutations(np.arange(1, p_param + 1))))
    p_fac = p_list.shape[0]

    anti_proj = sparse.lil_matrix((dimp, dimp))
    for j in range(p_fac):
        anti_proj += perm_sign(p_list[j, :]) * permutation_operator(
            dim * np.ones(p_param), p_list[j, :], False, True
        )
    anti_proj = anti_proj / p_fac

    if partial:
        anti_proj = anti_proj.todense()
        anti_proj = sparse.lil_matrix(linalg.orth(anti_proj))
    return anti_proj
