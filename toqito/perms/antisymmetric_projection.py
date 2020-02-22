"""Produces the projection onto the antisymmetric subspace."""
import numpy as np
import scipy as sp
from itertools import permutations
from toqito.perms.permutation_operator import permutation_operator
from toqito.perms.perm_sign import perm_sign


def antisymmetric_projection(dim: int,
                             p: int = 2,
                             partial: bool = False) -> sp.sparse.lil_matrix:
    """
    Produce the projection onto the antisymmetric subspace.

    Produces the orthogonal projection onto the anitsymmetric subspace of `p`
    copies of `dim`-dimensional space. If `partial = True`, then the
    antisymmetric projection (PA) isn't the orthogonal projection itself, but
    rather a matrix whose columns form an orthonormal basis for the symmetric
    subspace (and hence the PA * PA' is the orthogonal projection onto the
    symmetric subspace.)

    :param dim: The dimension of the local systems.
    :param p: Default value of 2.
    :param partial: Default value of 0.
    :return: Projection onto the antisymmetric subspace.
    """
    dimp = dim**p
    
    if p == 1:
        return sp.sparse.eye(dim)
    # The antisymmetric subspace is empty if `dim < p`.
    if dim < p:
        return sp.sparse.lil_matrix((dimp, dimp*(1-partial)))
    
    p_list = np.array(list(permutations(np.arange(1, p+1))))
    p_fac = p_list.shape[0]

    PA = sp.sparse.lil_matrix((dimp, dimp))
    for j in range(p_fac):
        PA += perm_sign(p_list[j, :]) * permutation_operator(dim*np.ones(p), p_list[j, :], False, True)
    PA = PA/p_fac

    if partial:
        PA = PA.todense()
        PA = sp.sparse.lil_matrix(sp.linalg.orth(PA))
    return PA

