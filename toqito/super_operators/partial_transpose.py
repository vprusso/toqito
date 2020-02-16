"""Computes the partial transpose of a matrix."""
from typing import List, Union
import numpy as np
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
    sqrt_rho_dims = np.round(np.sqrt(list(rho.shape)))

    if dim is None:
        dim = np.array([[sqrt_rho_dims[0], sqrt_rho_dims[0]],
                        [sqrt_rho_dims[1], sqrt_rho_dims[1]]])
    if isinstance(dim, float):
        dim = np.array([dim])

    num_sys = len(dim)

    # Allow the user to enter a single number for dim.
    if num_sys == 1:
        dim = np.array([dim, list(rho.shape)[0]/dim])
        if np.abs(dim[1] - np.round(dim[1])) >= 2*list(rho.shape)[0]*np.finfo(float).eps:
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
    sub_sys_vec_r = np.prod(dim[0][:]) * \
        np.ones(int(np.prod(dim[0][sys-1]))) / \
        np.prod(dim[0][sys-1])

    sub_sys_vec_c = np.prod(dim[1][:]) * \
        np.ones(int(np.prod(dim[1][sys-1]))) / \
        np.prod(dim[1][sys-1])

    set_diff = list(set(list(range(1, num_sys+1))) - {sys})

    perm = [sys]
    perm.extend(set_diff)

    # Permute the subsystems so that we just have to do the partial transpose
    # on the first (potentially larger) subsystem.
    rho_permuted = permute_systems(rho, perm, dim)

    # The `view_as_blocks` function behaves in a similar manner to the
    # "cell2mat" function provided from MATLAB.
    block_shape_x = int(sub_sys_vec_r[0])
    block_shape_y = int(sub_sys_vec_c[0])
    blocks = view_as_blocks(rho_permuted,
                            block_shape=(block_shape_x, block_shape_y))
    rho_permuted = np.hstack([np.vstack(block) for block in blocks])

    # Return the subsystems back to their original positions.
    dim[:, sys-1] = dim[[1, 0], sys-1]
    perm_np = np.array(perm)
    perm_np = list(perm_np - 1)
    dim = dim[:, perm_np]

    return permute_systems(rho_permuted, perm, dim, False, True)
