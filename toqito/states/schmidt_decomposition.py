"""Computes the Schmidt decomposition of a bipartite vector."""
from typing import List, Union
from scipy.sparse.linalg import svds
from scipy.linalg import svd

import numpy as np
import scipy as sp


def schmidt_decomposition(vec: np.ndarray,
                          dim: Union[int, List[int]] = None,
                          k_param: int = 0) -> np.ndarray:
    """
    Compute the Schmidt decomposition of a bipartite vector.

    """
    eps = np.finfo(float).eps

    if dim is None:
        dim = np.round(np.sqrt(len(vec)))

    # Allow the user to enter a single number for `dim`.
    if isinstance(dim, float):
        dim = np.array([dim, len(vec)/dim])
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * len(vec) * eps:
            msg = """
            """
            raise ValueError(msg)
        dim[1] = np.round(dim[1])

    # Try to gues whether SVD or SVDS will be faster, and then perform the
    # appropriate singular value decomposition.
    adj = 20 + 1000 * (not sp.sparse.issparse(vec))

    # Just a few Schmidt coefficients.
    if 0 < k_param <= np.ceil(np.min(dim) / adj):
        u_mat, singular_vals, vt_mat = svds(np.reshape(vec, dim[::-1]), k_param)
    # Otherwise, use lots of Schmidt coefficients.
    else:
        u_mat, singular_vals, vt_mat = svd(np.reshape(vec, dim[::-1].astype(int)))
        if k_param > 0:
            pass
