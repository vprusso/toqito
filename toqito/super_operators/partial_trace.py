import numpy as np
from scipy.sparse import issparse, csr_matrix
from skimage.util.shape import view_as_blocks
from toqito.perms.permute_systems import permute_systems
from typing import Any


def partial_trace(X: np.ndarray,
                  sys: Any = None,
                  dim: int = None,
                  mode: int = None):
    """
    Computes the partial trace of a matrix.

    :param X: A square matrix.
    :param sys:
    :param dim:
    :param mode:

    Gives the partial trace of the matrix X, where the dimensions of the
    (possibly more than 2) subsystems are given by the vector DIM and the
    subsystems to take the trace on are given by the scalar or vector SYS.

    MODE is a flag that determines which of two algorithms is used to compute
    the partial trace.

    If MODE = -1, then this script chooses whichever algorithm it thinks will
    be faster based on the dimensions of the subsystems being traced out and
    the sparisity of X. If you wish to force one specific algorithm, set either
    MODE = 0 (which generally works best for full or non- numeric matrices, or
    sparse matrices when most of the subsystems are being trace out) or MODE =
    1 (which generally works best when X is large and sparse, and the partial
    trace of X will also be large).
    """
    eps = np.finfo(float).eps
    lX = len(X)

    if dim is None:
        dim = np.array([np.round(np.sqrt(lX))])
    if mode is None:
        mode = -1

    if sys is None:
        sys = 2
    
    num_sys = len(dim)

    # Allow the user to enter a single number for dim.
    if num_sys == 1:
        dim = np.array([dim[0], lX/dim[0]])
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * lX * eps:
            msg = """
                InvalidDim: If DIM is a scalar, DIM must evenly divide length(X).
            """
            raise ValueError(msg)
        dim[1] = np.round(dim[1])
        num_sys = 2

    is_sparse = issparse(X)
    prod_dim = np.prod(dim)
    if isinstance(sys, list):
        prod_dim_sys = np.prod(dim)
    elif isinstance(sys, int):
        prod_dim_sys = np.prod(dim[sys-1])
    else:
        raise ValueError("ERROR")

    sub_sys_vec = prod_dim * np.ones(int(prod_dim_sys))//prod_dim_sys

    s1 = list(range(1, num_sys+1))
    if isinstance(sys, list):
        s2 = sys
    elif isinstance(sys, int):
        s2 = [sys]
    else:
        raise ValueError("ERROR")
    set_diff = list(set(s1) - set(s2))
   
    if isinstance(sys, list):
        perm = sys
    elif isinstance(sys, int):
        perm = [sys]
    else:
        raise ValueError("ERROR")
    perm.extend(set_diff)

    A = permute_systems(X, perm, dim)

    # Convert the elements of sub_sys_vec to integers and 
    # convert from a numpy array to a tuple to feed it into
    # the view_as_blocks function.
    X = view_as_blocks(A, block_shape=(int(sub_sys_vec[0]), int(sub_sys_vec[1])))

    Xpt = csr_matrix((int(sub_sys_vec[0]), int(sub_sys_vec[0])))
    
    for i in range(int(prod_dim_sys)):
        Xpt = Xpt + X[i, i]

    return Xpt

