import numpy as np
import operator
import functools

from toqito.matrix.operations.vec import vec


def permute_systems(X, perm, dim=None, row_only: bool=False, inv_perm: bool=False):
    if len(X.shape) == 1:
        dX = (1, X.shape[0])
    else:
        dX = X.shape

    is_vec = X.ndim == 1
    num_sys = len(perm)
    if is_vec:
        # 1 if column vector
        if len(dX) == 2:
            vec_orien = 1
        # 2 if row vector
        else:
            vec_orien = 0

    if dim is None:
        x = dX[0]**(1/num_sys) * np.ones(num_sys)
        y = dX[1]**(1/num_sys) * np.ones(num_sys)
        dim = np.array([x, y])

    if len(dim.shape) == 1:
        # Force dim to be a row vector.
        dim_tmp = dim[:].T
        if is_vec:
            dim = np.ones((2, len(dim)))
            dim[vec_orien, :] = dim_tmp
        else:
            dim = np.array([[dim_tmp],
                            [dim_tmp]])

    prod_dimR = np.prod(dim[0, :])
    prod_dimC = np.prod(dim[1, :])

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
            PX_1 = X.reshape(dim[vec_orien, ::-1].astype(int), order="F")
            PX = vec(np.transpose(PX_1, num_sys - np.array(perm[::-1]))).T
            # We need to flatten out the array.
            PX = functools.reduce(operator.iconcat, PX, [])
        else:
            PX_1 = X.reshape(dim[vec_orien, ::-1].astype(int), order="F")
            PX = vec(np.transpose(PX_1, num_sys - np.array(perm[::-1]))).T
            # We need to flatten out the array.
            PX = functools.reduce(operator.iconcat, PX, [])
        return PX
    
    vec_arg = np.array(list(range(0, dX[0])))

    row_perm = permute_systems(vec_arg, perm, dim[0, :], False, inv_perm)
    PX = X[row_perm, :]

    if not row_only:
        vec_arg = np.array(list(range(0, dX[1])))
        col_perm = permute_systems(vec_arg, perm, dim[1, :], False, inv_perm)
        PX = PX[:, col_perm]

    return PX

