"""Computes the Schmidt decomposition of a bipartite vector."""
from typing import List, Tuple, Union
from scipy.sparse.linalg import LinearOperator

import numpy as np
import scipy as sp


def schmidt_decomposition(vec: np.ndarray,
                          dim: Union[int, List[int]] = None,
                          k_param: int = 0) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Schmidt decomposition of a bipartite vector.

    :param vec:
    :param dim:
    :param k_param:
    :return: The Schmidt decomposition of the `vec` input.
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

    # Try to guess whether SVD or SVDS will be faster, and then perform the
    # appropriate singular value decomposition.
    adj = 20 + 1000 * (not sp.sparse.issparse(vec))

    # Just a few Schmidt coefficients.
    if 0 < k_param <= np.ceil(np.min(dim) / adj):
        u_mat, singular_vals, vt_mat = \
            sp.sparse.linalg.svds(LinearOperator(np.reshape(vec, dim[::-1]),
                                                 k_param))
    # Otherwise, use lots of Schmidt coefficients.
    else:
        u_mat, singular_vals, vt_mat = \
            np.linalg.svd(np.reshape(vec, dim[::-1].astype(int)))
    if k_param > 0:
        u_mat = u_mat[:, :k_param]
        singular_vals = singular_vals[:, :k_param]
        vt_mat = vt_mat[:, :k_param]

    #singular_vals = np.diag(singular_vals)
    if k_param == 0:
        # Schmidt rank.
        r_param = np.sum(
            singular_vals > np.max(dim) * np.spacing(singular_vals[0])
        )
        # Schmidt coefficients.
        singular_vals = singular_vals[:r_param]
        u_mat = u_mat[:, :r_param]
        vt_mat = vt_mat[:, :r_param]

    u_mat = u_mat.conj().T
    return singular_vals, u_mat, vt_mat
