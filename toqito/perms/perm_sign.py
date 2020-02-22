"""Computes the sign of a permutation."""
import numpy as np
import scipy as sp


def perm_sign(perm: np.ndarray) -> float:
    """
    Compute the "sign" of a permutation.

    The sign (either -1 or 1) of the permutation `perm` is -1**`inv`, where
    `inv` is the number of inversions contained in `perm`
    """
    iden = np.eye(len(perm))
    return sp.linalg.det(iden[:, np.array(perm)-1])
