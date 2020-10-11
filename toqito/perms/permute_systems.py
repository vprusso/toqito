"""Permute systems."""
import functools
import operator

from typing import List, Union
from scipy import sparse

import numpy as np

from toqito.matrix_ops import vec


def permute_systems(
    input_mat: np.ndarray,
    perm: Union[np.ndarray, List[int]],
    dim: Union[np.ndarray, List[int]] = None,
    row_only: bool = False,
    inv_perm: bool = False,
) -> np.ndarray:
    r"""
    Permute subsystems within a state or operator.

    Permutes the order of the subsystems of the vector or matrix :code:`input_mat` according to the
    permutation vector :code:`perm`, where the dimensions of the subsystems are given by the vector
    :code:`dim`. If :code:`input_mat` is non-square and not a vector, different row and column
    dimensions can be specified by putting the row dimensions in the first row of :code:`dim` and
    the columns dimensions in the second row of :code:`dim`.

    If :code:`row_only = True`, then only the rows of :code:`input_mat` are permuted, but not the
    columns -- this is equivalent to multiplying :code:`input_mat` on the left by the corresponding
    permutation operator, but not on the right.

    If :code:`row_only = False`, then :code:`dim` only needs to contain the row dimensions of the
    subsystems, even if :code:`input_mat` is not square. If :code:`inv_perm = True`, then the
    inverse permutation of :code:`perm` is applied instead of :code:`perm` itself.

    Examples
    ==========

    For spaces :math:`\mathcal{A}` and :math:`\mathcal{B}` where
    :math:`\text{dim}(\mathcal{A}) = \text{dim}(\mathcal{B}) = 2` we may consider an operator
    :math:`X \in \mathcal{A} \otimes \mathcal{B}`. Applying the `permute_systems` function with
    vector :math:`[2,1]` on :math:`X`, we may reorient the spaces such that
    :math:`X \in \mathcal{B} \otimes \mathcal{A}`.

    For example, if we define :math:`X \in \mathcal{A} \otimes \mathcal{B}` as

    .. math::
        X = \begin{pmatrix}
            1 & 2 & 3 & 4 \\
            5 & 6 & 7 & 8 \\
            9 & 10 & 11 & 12 \\
            13 & 14 & 15 & 16
        \end{pmatrix},

    then applying the `permute_systems` function on :math:`X` to obtain
    :math:`X \in \mathcal{B} \otimes \mathcal{A}` yield the following matrix

    .. math::
        X_{[2,1]} = \begin{pmatrix}
            1 & 3 & 2 & 4 \\
            9 & 11 & 10 & 12 \\
            5 & 7 & 6 & 8 \\
            13 & 15 & 14 & 16
        \end{pmatrix}.

    >>> from toqito.perms import permute_systems
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> )
    >>> permute_systems(test_input_mat, [2, 1])
    [[ 1  3  2  4]
     [ 9 11 10 12]
     [ 5  7  6  8]
     [13 15 14 16]]

    For spaces :math:`\mathcal{A}, \mathcal{B}`, and :math:`\mathcal{C}` where
    :math:`\text{dim}(\mathcal{A}) = \text{dim}(\mathcal{B}) = \text{dim}(\mathcal{C}) = 2` we may
    consider an operator :math:`X \in \mathcal{A} \otimes \mathcal{B} \otimes \mathcal{C}`. Applying
    the :code:`permute_systems` function with vector :math:`[2,3,1]` on :math:`X`, we may reorient
    the spaces such that :math:`X \in \mathcal{B} \otimes \mathcal{C} \otimes \mathcal{A}`.

    For example, if we define :math:`X \in \mathcal{A} \otimes \mathcal{B} \otimes \mathcal{C}` as

    .. math::
        X =
        \begin{pmatrix}
            1 & 2 & 3 & 4, 5 & 6 & 7 & 8 \\
            9 & 10 & 11 & 12 & 13 & 14 & 15 & 16 \\
            17 & 18 & 19 & 20 & 21 & 22 & 23 & 24 \\
            25 & 26 & 27 & 28 & 29 & 30 & 31 & 32 \\
            33 & 34 & 35 & 36 & 37 & 38 & 39 & 40 \\
            41 & 42 & 43 & 44 & 45 & 46 & 47 & 48 \\
            49 & 50 & 51 & 52 & 53 & 54 & 55 & 56 \\
            57 & 58 & 59 & 60 & 61 & 62 & 63 & 64
        \end{pmatrix},

    then applying the `permute_systems` function on :math:`X` to obtain
    :math:`X \in \mathcal{B} \otimes \mathcal{C} \otimes \mathcal{C}` yield the following matrix

    .. math::
        X_{[2, 3, 1]} =
        \begin{pmatrix}
            1 & 5 & 2 & 6 & 3 & 7 & 4, 8 \\
            33 & 37 & 34 & 38 & 35 & 39 & 36 & 40 \\
            9 & 13 & 10 & 14 & 11 & 15 & 12 & 16 \\
            41 & 45 & 42 & 46 & 43 & 47 & 44 & 48 \\
            17 & 21 & 18 & 22 & 19 & 23 & 20 & 24 \\
            49 & 53 & 50 & 54 & 51 & 55 & 52 & 56 \\
            25 & 29 & 26 & 30 & 27 & 31 & 28 & 32 \\
            57 & 61 & 58 & 62 & 59 & 63 & 60 & 64
        \end{pmatrix}.

    >>> from toqito.perms import permute_systems
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>    [
    >>>        [1, 2, 3, 4, 5, 6, 7, 8],
    >>>        [9, 10, 11, 12, 13, 14, 15, 16],
    >>>        [17, 18, 19, 20, 21, 22, 23, 24],
    >>>        [25, 26, 27, 28, 29, 30, 31, 32],
    >>>        [33, 34, 35, 36, 37, 38, 39, 40],
    >>>        [41, 42, 43, 44, 45, 46, 47, 48],
    >>>        [49, 50, 51, 52, 53, 54, 55, 56],
    >>>        [57, 58, 59, 60, 61, 62, 63, 64],
    >>>    ]
    >>> )
    >>> permute_systems(test_input_mat, [2, 3, 1])
    [[ 1  5  2  6  3  7  4  8]
     [33 37 34 38 35 39 36 40]
     [ 9 13 10 14 11 15 12 16]
     [41 45 42 46 43 47 44 48]
     [17 21 18 22 19 23 20 24]
     [49 53 50 54 51 55 52 56]
     [25 29 26 30 27 31 28 32]
     [57 61 58 62 59 63 60 64]]

    :param input_mat: The vector or matrix.
    :param perm: A permutation vector.
    :param dim: The default has all subsystems of equal dimension.
    :param row_only: Default: :code:`False`
    :param inv_perm: Default: :code:`True`
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

    if isinstance(dim, list):
        dim = np.array(dim)

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
    if input_mat_dims[0] != prod_dim_r or (not row_only and input_mat_dims[1] != prod_dim_c):
        raise ValueError(
            "InvalidDim: The dimensions specified in DIM do not " "agree with the size of X."
        )
    if is_vec:
        if inv_perm:
            permuted_mat_1 = input_mat.reshape(dim[vec_orien, ::-1].astype(int), order="F")
            permuted_mat = vec(np.transpose(permuted_mat_1, num_sys - np.array(perm[::-1]))).T
            # We need to flatten out the array.
            permuted_mat = functools.reduce(operator.iconcat, permuted_mat, [])
        else:
            permuted_mat_1 = input_mat.reshape(dim[vec_orien, ::-1].astype(int), order="F")
            permuted_mat = vec(np.transpose(permuted_mat_1, num_sys - np.array(perm[::-1]))).T
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
