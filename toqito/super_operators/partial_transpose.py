import numpy as np
from typing import List, Union
from skimage.util.shape import view_as_blocks
from toqito.perms.permute_systems import permute_systems


def partial_transpose(rho: np.ndarray,
                      sys: Union[List[int], np.ndarray, int] = 2,
                      dim: Union[List[int], np.ndarray] = None) -> np.ndarray:
    """
    Compute the partial transpose of a matrix.

    :param rho: A matrix.
    :param sys: Scalar or vector specifying the size of the subsystems.
    :param dim: Dimension of the subsystems. If `None`, all dimensions
                are assumed to be equal.
    :returns: The partial transpose of matrix `rho`.

    By default, the returned matrix is the partial transpose of the matrix
    `rho`, where it is assumed that the number of rows and columns of `rho` are
    both perfect squares and both subsystems have equal dimension. The
    transpose is applied to the second subsystem.

    In the case where `sys` amd `dim` are specified, this function gives the
    partial transpose of the matrix `rho` where the dimensions of the (possibly
    more than 2) subsystems are given by the vector `dim` and the subsystems to
    take the partial transpose are given by the scalar or vector `sys`. If
    `rho` is non-square, different row and column dimensions can be specified
    by putting the row dimensions in the first row of `dim` and the column
    dimensions in the second row of `dim`.
    """
    eps = np.finfo(float).eps
    rho_dims = list(rho.shape)
    sqrt_rho_dims = np.round(np.sqrt(rho_dims))

    if dim is None:
        dim = np.array([[sqrt_rho_dims[0], sqrt_rho_dims[0]],
                        [sqrt_rho_dims[1], sqrt_rho_dims[1]]])
    if isinstance(dim, float):
        dim = np.array([dim])

    num_sys = len(dim)

    # Allow the user to enter a single number for dim.
    if num_sys == 1:
        dim = np.array([dim, rho_dims[0]/dim])
        if np.abs(dim[1] - np.round(dim[1])) >= 2*rho_dims[0]*eps:
            msg = """
                InvalidDim: If `dim` is a scalar, `rho` must be square and
                `dim` must evenly divide `len(rho)`; please provide the `dim`
                array containing the dimensions of the subsystems.
            """
            raise ValueError(msg)
        dim[1] = np.round(dim[1])
        num_sys = 2

    # Allow the user to enter a vector for dim if X is square.
    if min(dim.shape) == 1 or len(dim.shape) == 1:
        # Force dim to be a row vector.
        dim = dim.T.flatten()
        dim = np.array([dim, dim])

    # Prepare the partial transposition.
    prod_dimR = np.prod(dim[0][:])
    prod_dimC = np.prod(dim[1][:])
    sub_prodR = np.prod(dim[0][sys-1])
    sub_prodC = np.prod(dim[1][sys-1])
    sub_sys_vecR = prod_dimR * np.ones(int(sub_prodR)) / sub_prodR
    sub_sys_vecC = prod_dimC * np.ones(int(sub_prodC)) / sub_prodC

    s1 = list(range(1, num_sys+1))
    s2 = [sys]
    set_diff = list(set(s1) - set(s2))

    perm = [sys]
    perm.extend(set_diff)

    # Permute the subsystems so that we just have to do the partial transpose
    # on the first (potentially larger) subsystem.
    rho_permuted = permute_systems(rho, perm, dim)

    # The `view_as_blocks` function behaves in a similar manner to the
    # "cell2mat" function provided from MATLAB.
    block_shape_x = int(sub_sys_vecR[0])
    block_shape_y = int(sub_sys_vecC[0])
    blocks = view_as_blocks(rho_permuted,
                            block_shape=(block_shape_x, block_shape_y))
    rho_permuted = np.hstack([np.vstack(block) for block in blocks])

    # Return the subsystems back to their original positions.
    dim[:, sys-1] = dim[[1, 0], sys-1]
    perm_np = np.array(perm)
    perm_np = list(perm_np - 1)
    dim = dim[:][perm_np]

    return permute_systems(rho_permuted, perm, dim, False, True)

