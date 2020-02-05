import numpy as np
import scipy as sp
import operator
import functools

from typing import List
from toqito.matrix.operations.vec import vec


def permute_systems(input_mat: np.ndarray,
                    perm: List[int],
                    dim=None,
                    row_only: bool = False,
                    inv_perm: bool = False) -> np.ndarray:
    """
    """
    if len(input_mat.shape) == 1:
        dX = (1, input_mat.shape[0])
    else:
        dX = input_mat.shape

    is_vec = np.min(dX) == 1
    num_sys = len(perm)

    if dim is None:
        x = dX[0]**(1/num_sys) * np.ones(num_sys)
        y = dX[1]**(1/num_sys) * np.ones(num_sys)
        dim = np.array([x, y])

    if is_vec:
        # 1 if column vector
        if len(input_mat.shape) > 1:
            vec_orien = 0
        # 2 if row vector
        elif len(input_mat.shape) == 1:
            vec_orien = 1
        else:
            msg = """
                InvalidMat: Length of tuple of dimensions specifying the input 
                matrix can only be of length 1 or length 2.
            """
            raise ValueError(msg)
    
    if len(dim.shape) == 1:
        # Force dim to be a row vector.
        dim_tmp = dim[:].T
        if is_vec:
            dim = np.ones((2, len(dim)))
            print(dim)
            dim[vec_orien, :] = dim_tmp
        else:
            dim = np.array([[dim_tmp],
                            [dim_tmp]])

    prod_dimR = int(np.prod(dim[0, :]))
    prod_dimC = int(np.prod(dim[1, :]))

    if len(perm) != num_sys:
        msg = """
            InvalidPerm: length(PERM) must be equal to length(DIM).
        """
        raise ValueError(msg)
    elif sorted(perm) != list(range(1, num_sys+1)):
        msg = """
            InvalidPerm: PERM must be a permutation vector.
        """
        raise ValueError(msg)
    elif dX[0] != prod_dimR or (not row_only and dX[1] != prod_dimC):
        msg = """
            InvalidDim: The dimensions specified in DIM do not agree with
            the size of X.
        """
        raise ValueError(msg)
    if is_vec:
        if inv_perm:
            PX_1 = input_mat.reshape(dim[vec_orien, ::-1].astype(int), order="F")
            PX = vec(np.transpose(PX_1, num_sys - np.array(perm[::-1]))).T
            # We need to flatten out the array.
            PX = functools.reduce(operator.iconcat, PX, [])
        else:
            PX_1 = input_mat.reshape(dim[vec_orien, ::-1].astype(int), order="F")
            PX = vec(np.transpose(PX_1, num_sys - np.array(perm[::-1]))).T
            # We need to flatten out the array.
            PX = functools.reduce(operator.iconcat, PX, [])
        return np.array(PX)
    
    vec_arg = np.array(list(range(0, dX[0])))

    # If the dimensions are specified, ensure they are given to the
    # recursive calls as flattened lists.
    if len(dim[0][:]) == 1:
        dim = functools.reduce(operator.iconcat, dim, [])

    row_perm = permute_systems(vec_arg, perm, dim[0][:], False, inv_perm)
    
    # This condition is only necessary if the `input_mat` variable is sparse.
    if isinstance(input_mat, sp.sparse.dia_matrix) or \
        isinstance(input_mat, sp.sparse.csr_matrix):
        input_mat = input_mat.toarray()
        PX = input_mat[row_perm, :]
        PX = np.array(PX)
    else:
        PX = input_mat[row_perm, :]

    if not row_only:
        vec_arg = np.array(list(range(0, dX[1])))
        col_perm = permute_systems(vec_arg, perm, dim[1][:], False, inv_perm)
        PX = PX[:, col_perm]
    
    return PX

