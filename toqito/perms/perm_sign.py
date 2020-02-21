"""Computes the sign of a permutation."""
import scipy as sp


def perm_sign(perm):
    """
    """
    iden = sp.sparse.eye(len(perm))
    print(iden[:, perm])
    #return sp.linalg.det(iden[:, perm])
