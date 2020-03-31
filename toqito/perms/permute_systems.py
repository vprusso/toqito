"""Permutes subsystems within a state or operator."""
from typing import List
from scipy import sparse
import functools
import operator
import numpy as np

from toqito.matrix.operations.vec import vec


def permute_systems(
    input_mat: np.ndarray,
    perm: List[int],
    dim=None,
    row_only: bool = False,
    inv_perm: bool = False,
) -> np.ndarray:
    """
    Permute subsystems within a state or operator.

    Permutes the order of the subsystems of the vector or matrix `input_mat`
    according to the permutation vector `perm`, where the dimensions of the
    subsystems are given by the vector `dim`. If `input_mat` is non-square and
    not a vector, different row and column dimensions can be specified by
    putting the row dimensions in the first row of `dim` and the columns
    dimensions in the second row of `dim`.

    If `row_only = True`, then only the rows of `input_mat` are permuted, but
    not the columns -- this is equivalent to multiplying `input_mat` on the
    left by the corresponding permutation operator, but not on the right.

    If `row_only = False`, then `dim` only needs to contain the row dimensions
    of the subsystems, even if `input_mat` is not square. If `inv_perm = True`,
    then the inverse permutation of `perm` is applied instead of `perm` itself.

    :param input_mat: The vector or matrix.
    :param perm: A permutation vector.
    :param dim: The default has all subsystems of equal dimension.
    :param row_only: Default: `False`
    :param inv_perm: Default :`True`
    :return: The matrix or vector that has been permuted.
    """
    if len(input_mat.shape) == 1:
        input_mat_dims = (1, input_mat.shape[0])
    else:
        input_mat_dims = input_mat.shape

    is_vec = np.min(input_mat_dims) == 1
    num_sys = len(perm)

    if dim is None:
        x_tmp = input_mat_dims[0] ** (1 / num_sys) * np.ones(num_sys)
        y_tmp = input_mat_dims[1] ** (1 / num_sys) * np.ones(num_sys)
        dim = np.array([x_tmp, y_tmp])

    if is_vec:
        # 1 if column vector
        if len(input_mat.shape) > 1:
            vec_orien = 0
        # 2 if row vector
        elif len(input_mat.shape) == 1:
            vec_orien = 1
        else:
            raise ValueError(
                "InvalidMat: Length of tuple of dimensions "
                "specifying the input matrix can only be of "
                "length 1 or length 2."
            )

    if len(dim.shape) == 1:
        # Force dim to be a row vector.
        dim_tmp = dim[:].T
        if is_vec:
            dim = np.ones((2, len(dim)))
            dim[vec_orien, :] = dim_tmp
        else:
            dim = np.array([[dim_tmp], [dim_tmp]])

    prod_dim_r = int(np.prod(dim[0, :]))
    prod_dim_c = int(np.prod(dim[1, :]))

    if len(perm) != num_sys:
        raise ValueError("InvalidPerm: `len(perm)` must be equal to " "`len(dim)`.")
    if sorted(perm) != list(range(1, num_sys + 1)):
        raise ValueError("InvalidPerm: `perm` must be a permutation vector.")
    if input_mat_dims[0] != prod_dim_r or (
        not row_only and input_mat_dims[1] != prod_dim_c
    ):
        raise ValueError(
            "InvalidDim: The dimensions specified in DIM do not "
            "agree with the size of X."
        )
    if is_vec:
        if inv_perm:
            permuted_mat_1 = input_mat.reshape(
                dim[vec_orien, ::-1].astype(int), order="F"
            )
            permuted_mat = vec(
                np.transpose(permuted_mat_1, num_sys - np.array(perm[::-1]))
            ).T
            # We need to flatten out the array.
            permuted_mat = functools.reduce(operator.iconcat, permuted_mat, [])
        else:
            permuted_mat_1 = input_mat.reshape(
                dim[vec_orien, ::-1].astype(int), order="F"
            )
            permuted_mat = vec(
                np.transpose(permuted_mat_1, num_sys - np.array(perm[::-1]))
            ).T
            # We need to flatten out the array.
            permuted_mat = functools.reduce(operator.iconcat, permuted_mat, [])
        return np.array(permuted_mat)

    vec_arg = np.array(list(range(0, input_mat_dims[0])))

    # If the dimensions are specified, ensure they are given to the
    # recursive calls as flattened lists.
    if len(dim[0][:]) == 1:
        dim = functools.reduce(operator.iconcat, dim, [])

    row_perm = permute_systems(vec_arg, perm, dim[0][:], False, inv_perm)

    # This condition is only necessary if the `input_mat` variable is sparse.
    if isinstance(input_mat, (sparse.csr_matrix, sparse.dia_matrix)):
        input_mat = input_mat.toarray()
        permuted_mat = input_mat[row_perm, :]
        permuted_mat = np.array(permuted_mat)
    else:
        permuted_mat = input_mat[row_perm, :]

    if not row_only:
        vec_arg = np.array(list(range(0, input_mat_dims[1])))
        col_perm = permute_systems(vec_arg, perm, dim[1][:], False, inv_perm)
        permuted_mat = permuted_mat[:, col_perm]

    return permuted_mat
