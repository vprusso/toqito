import numpy as np
from scipy.sparse import issparse
from toqito.helper.permute_systems import permute_systems


def partial_trace(X: np.ndarray,
                  sys: int = None,
                  dim: int = None,
                  mode: int = None):

    eps = np.finfo(float).eps
    lX = len(X)

    if sys is None:
        sys = 2
    if dim is None:
        dim = np.array([np.round(np.sqrt(lX))])
    if mode is None:
        mode = -1
    
    num_sys = len(dim)

    # Allow the user to enter a single number for dim.
    if num_sys == 1:
        dim = np.array([dim[0], lX/dim[0]])
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * lX * eps:
            raise ValueError("InvalidDim:")
        dim[1] = np.round(dim[1])
        num_sys = 2

    is_sparse = issparse(X)
    prod_dim = np.prod(dim)
    prod_dim_sys = np.prod(dim[sys-1])

    sub_sys_vec = prod_dim * np.ones(int(prod_dim_sys))/prod_dim_sys

    perm = [2, 1]
    Y = permute_systems(X, [2, 1])

