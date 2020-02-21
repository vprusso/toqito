"""Produces the projection onto the symmetric subspace."""
import numpy as np
import scipy as sp
from itertools import permutations
from toqito.perms.permutation_operator import permutation_operator


def symmetric_projection(dim: int,
                         p: int = 2,
                         partial: bool = False) -> np.ndarray:
    """
    Produce the projection onto the symmetric subspace.

    Produces the orthogonal projection onto the symmetric subspace of `p`
    copies of `dim`-dimensional space. If `partail = True`, then the symmetric
    projection (PS) isn't the orthogonal projection itself, but rather a matrix
    whose columns form an orthonormal basis for the symmetric subspace (and
    hence the PS * PS' is the orthogonal projection onto the symmetric
    subspace.)

    :param dim: The dimension of the local systems.
    :param p: Default value of 2.
    :param partial: Default value of 0.
    :return: Projection onto the symmetric subspace.
    """
    dimp = dim**p
    
    if p == 1:
        return sp.sparse.eye(dim)

    p_list = np.array(list(permutations(np.arange(1, p+1))))
    p_fac = np.math.factorial(p)
    PS = sp.sparse.lil_matrix((dimp, dimp))

    for j in range(p_fac):
        PS += permutation_operator(dim*np.ones(p), p_list[j, :], False, True)
    PS = PS/p_fac

    if partial:
        PS = PS.todense()
        PS = sp.sparse.lil_matrix(sp.linalg.orth(PS))
    return PS

