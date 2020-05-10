"""Permutation and combinatorial functions."""
import functools
import operator

from typing import List, Union

from dataclasses import dataclass
from itertools import permutations
from scipy import linalg, sparse

import numpy as np

from toqito.matrices import iden
from toqito.matrix_ops import vec


__all__ = [
    "antisymmetric_projection",
    "perm_sign",
    "permutation_operator",
    "permute_systems",
    "swap",
    "swap_operator",
    "symmetric_projection",
    "unique_perms",
]


def antisymmetric_projection(
    dim: int, p_param: int = 2, partial: bool = False
) -> sparse.lil_matrix:
    r"""
    Produce the projection onto the antisymmetric subspace [WikAsym]_.

    Produces the orthogonal projection onto the anti-symmetric subspace of `p`
    copies of `dim`-dimensional space. If `partial = True`, then the
    antisymmetric projection (PA) isn't the orthogonal projection itself, but
    rather a matrix whose columns form an orthonormal basis for the symmetric
    subspace (and hence the PA * PA' is the orthogonal projection onto the
    symmetric subspace.)

    Examples
    ==========

    The :math:`2`-dimensional antisymmetric projection with :math:`p=1` is given
    as :math:`2`-by-:math:`2` identity matrix

    .. math::
        \begin{pmatrix}
            1 & 0 \\
            0 & 1
        \end{pmatrix}.

    Using `toqito`, we can see this gives the proper result.

    >>> from toqito.perms import antisymmetric_projection
    >>> antisymmetric_projection(2, 1).todense()
    [[1., 0.],
     [0., 1.]]

    When the :math:`p` value is greater than the dimension of the antisymmetric
    projection, this just gives the matrix consisting of all zero entries. For
    instance, when :math:`d = 2` and :math:`p = 3` we have that

    .. math::
        \begin{pmatrix}
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
        \end{pmatrix}.

    Using `toqito` we can see this gives the proper result.

    >>> from toqito.perms import antisymmetric_projection
    >>> antisymmetric_projection(2, 3).todense()
    [[0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0.]]

    References
    ==========
    .. [WikAsym] Wikipedia: Anti-symmetric operator
        https://en.wikipedia.org/wiki/Anti-symmetric_operator

    :param dim: The dimension of the local systems.
    :param p_param: Default value of 2.
    :param partial: Default value of 0.
    :return: Projection onto the antisymmetric subspace.
    """
    dimp = dim ** p_param

    if p_param == 1:
        return sparse.eye(dim)
    # The antisymmetric subspace is empty if `dim < p`.
    if dim < p_param:
        return sparse.lil_matrix((dimp, dimp * (1 - partial)))

    p_list = np.array(list(permutations(np.arange(1, p_param + 1))))
    p_fac = p_list.shape[0]

    anti_proj = sparse.lil_matrix((dimp, dimp))
    for j in range(p_fac):
        anti_proj += perm_sign(p_list[j, :]) * permutation_operator(
            dim * np.ones(p_param), p_list[j, :], False, True
        )
    anti_proj = anti_proj / p_fac

    if partial:
        anti_proj = anti_proj.todense()
        anti_proj = sparse.lil_matrix(linalg.orth(anti_proj))
    return anti_proj


def perm_sign(perm: Union[np.ndarray, List[int]]) -> float:
    """
    Compute the "sign" of a permutation [WikParPerm]_.

    The sign (either -1 or 1) of the permutation `perm` is -1**`inv`, where
    `inv` is the number of inversions contained in `perm`.

    Examples
    ==========

    For the following vector

    .. math::
        [1, 2, 3, 4]

    the permutation sign is positive as the number of elements in the vector are
    even. This can be performed in `toqito` as follows.

    >>> from toqito.perms import perm_sign
    >>> perm_sign([1, 2, 3, 4])
    1

    For the following vector

    .. math::
        [1, 2, 3, 4, 5]

    the permutation sign is negative as the number of elements in the vector are
    odd. This can be performed in `toqito` as follows.

    >>> from toqito.perms import perm_sign
    >>> perm_sign([1, 2, 4, 3, 5])
    -1

    References
    ==========
    .. [WikParPerm] Wikipedia: Parity of a permutation
        https://en.wikipedia.org/wiki/Parity_of_a_permutation

    :param perm: The permutation vector to be checked.
    :return: The value 1 if the permutation is of even length and the value of
             -1 if the permutation is of odd length.
    """
    eye = np.eye(len(perm))
    return linalg.det(eye[:, np.array(perm) - 1])


def permutation_operator(
    dim: Union[List[int], int],
    perm: List[int],
    inv_perm: bool = False,
    is_sparse: bool = False,
) -> np.ndarray:
    r"""
    Produce a unitary operator that permutes subsystems.

    Generates a unitary operator that permutes the order of subsystems
    according to the permutation vector `perm`, where the ith subsystem has
    dimension `dim[i]`.

    If `inv_perm` = True, it implements the inverse permutation of `perm`. The
    permutation operator return is full is `is_sparse` is False and sparse if
    `is_sparse` is True.

    Examples
    ==========

    The permutation operator obtained with dimension :math:`d = 2` is equivalent
    to the standard swap operator on two qubits

    .. math::
        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1
        \end{pmatrix}

    Using `toqito`, this can be achieved in the following manner.

    >>> from toqito.perms import permutation_operator
    >>> permutation_operator(2, [2, 1])
    [[1., 0., 0., 0.],
     [0., 0., 1., 0.],
     [0., 1., 0., 0.],
     [0., 0., 0., 1.]]

    :param dim: The dimensions of the subsystems to be permuted.
    :param perm: A permutation vector.
    :param inv_perm: Boolean dictating if `perm` is inverse or not.
    :param is_sparse: Boolean indicating if return is sparse or not.
    :return: Permutation operator of dimension `dim`.
    """
    # Allow the user to enter a single number for `dim`.
    if isinstance(dim, int):
        dim = dim * np.ones(max(perm))
    if isinstance(dim, list):
        dim = np.array(dim)

    # Swap the rows of the identity matrix appropriately.
    return permute_systems(
        iden(int(np.prod(dim)), is_sparse), perm, dim, True, inv_perm
    )


def permute_systems(
    input_mat: np.ndarray,
    perm: List[int],
    dim=None,
    row_only: bool = False,
    inv_perm: bool = False,
) -> np.ndarray:
    r"""
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

    Examples
    ==========

    For spaces :math:`\mathcal{A}` and :math:`\mathcal{B}` where
    :math:`\text{dim}(\mathcal{A}) = \text{dim}(\mathcal{B}) = 2` we may
    consider an operator :math:`X \in \mathcal{A} \otimes \mathcal{B}`. Applying
    the `permute_systems` function with vector :math:`[2,1]` on :math:`X`, we
    may reorient the spaces such that
    :math:`X \in \mathcal{B} \otimes \mathcal{A}`.

    For example, if we define :math:`X \in \mathcal{A} \otimes \mathcal{B}` as

    .. math::
        X = \begin{pmatrix}
            1 & 2 & 3 & 4 \\
            5 & 6 & 7 & 8 \\
            9 & 10 & 11 & 12 \\
            13 & 14 & 15 & 16
        \end{pmatrix}

    then applying the `permute_systems` function on :math:`X` to obtain
    :math:`X \in \mathcal{B} \otimes \mathcal{A}` yield the following matrix

    .. math::
        X_{[2,1]} = \begin{pmatrix}
            1 & 3 & 2 & 4 \\
            9 & 11 & 10 & 12 \\
            5 & 7 & 6 & 8 \\
            13 & 15 & 14 & 16
        \end{pmatrix}

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

    For spaces :math:`\mathcal{A}, \mathcal{B}`, and :math:`\mathcal{C}`
    where :math:`\text{dim}(\mathcal{A}) = \text{dim}(\mathcal{B}) =
    \text{dim}(\mathcal{C}) = 2` we may consider an operator
    :math:`X \in \mathcal{A} \otimes \mathcal{B} \otimes \mathcal{C}`. Applying
    the `permute_systems` function with vector :math:`[2,3,1]` on :math:`X`, we
    may reorient the spaces such that
    :math:`X \in \mathcal{B} \otimes \mathcal{C} \otimes \mathcal{A}`.

    For example, if we define
    :math:`X \in \mathcal{A} \otimes \mathcal{B} \otimes \mathcal{C}` as

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
        \end{pmatrix}

    then applying the `permute_systems` function on :math:`X` to obtain
    :math:`X \in \mathcal{B} \otimes \mathcal{C} \otimes \mathcal{C}` yield the
    following matrix

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
        \end{pmatrix}

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


def swap(
    rho: np.ndarray,
    sys: List[int] = None,
    dim: Union[List[int], List[List[int]], int, np.ndarray] = None,
    row_only: bool = False,
) -> np.ndarray:
    r"""
    Swap two subsystems within a state or operator.

    Swaps the two subsystems of the vector or matrix `rho`, where the
    dimensions of the (possibly more than 2) subsystems are given by `dim` and
    the indices of the two subsystems to be swapped are specified in the 1-by-2
    vector `sys`.

    If `rho` is non-square and not a vector, different row and column
    dimensions can be specified by putting the row dimensions in the first row
    of `dim` and the column dimensions in the second row of `dim`.

    If `row_only` is set to `True`, then only the rows of `rho` are swapped,
    but not the columns -- this is equivalent to multiplying `rho` on the left
    by the corresponding swap operator, but not on the right.

    Examples
    ==========

    Consider the following matrix

    .. math::
        X =
        \begin{pmatrix}
            1 & 5 & 9 & 13 \\
            2 & 6 & 10 & 14 \\
            3 & 7 & 11 & 15 \\
            4 & 8 & 12 & 16
        \end{pmatrix}

    If we apply the `swap` function provided by `toqito` on :math:`X`, we should
    obtain the following matrix

    .. math::
        \text{Swap}(X) =
        \begin{pmatrix}
            1 & 9 & 5 & 13 \\
            3 & 11 & 7 & 15 \\
            2 & 10 & 6 & 14 \\
            4 & 12 & 8 & 16
        \end{pmatrix}

    This can be observed by the following example in `toqito`.

    >>> from toqito.perms import swap
    >>> import numpy as np
    >>> test_mat = np.array(
    >>>     [[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]]
    >>> )
    >>> swap(test_mat)
    [[ 1  9  5 13]
     [ 3 11  7 15]
     [ 2 10  6 14]
     [ 4 12  8 16]]

    It is also possible to use the `sys` and `dim` arguments, it is possible to
    specify the system and dimension on which to apply the swap operator. For
    instance for `sys = [1 ,2]` and `dim = 2` we have that

    .. math::
        \begin{pmatrix}
            1 & 9 & 5 & 13 \\
            3 & 11 & 7 & 15 \\
            2 & 10 & 6 & 14 \\
            4 & 12 & 8 & 16
        \end{pmatrix}.

    Using `toqito` we can see this gives the proper result.

    >>> from toqito.perms import swap
    >>> import numpy as np
    >>> test_mat = np.array(
    >>>     [[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]]
    >>> )
    >>> swap(test_mat, [1, 2], 2)
    [[ 1  9  5 13]
     [ 3 11  7 15]
     [ 2 10  6 14]
     [ 4 12  8 16]]

    It is also possible to perform the `swap` function on vectors in addition to
    matrices.

    >>> from toqito.perms import swap
    >>> import numpy as np
    >>> test_vec = np.array([1, 2, 3, 4])
    >>> swap(test_vec)
    [1 3 2 4]

    :param rho: A vector or matrix to have its subsystems swapped.
    :param sys: Default: [1, 2]
    :param dim: Default: [sqrt(len(X), sqrt(len(X)))]
    :param row_only: Default: False
    :return: The swapped matrix.
    """
    eps = np.finfo(float).eps
    if len(rho.shape) == 1:
        rho_dims = (1, rho.shape[0])
    else:
        rho_dims = rho.shape

    round_dim = np.round(np.sqrt(rho_dims))

    if sys is None:
        sys = [1, 2]

    if isinstance(dim, list):
        dim = np.array(dim)
    if dim is None:
        dim = np.array([[round_dim[0], round_dim[0]], [round_dim[1], round_dim[1]]])

    if isinstance(dim, int):
        dim = np.array([[dim, rho_dims[0] / dim], [dim, rho_dims[1] / dim]])
        if (
            np.abs(dim[0, 1] - np.round(dim[0, 1]))
            + np.abs(dim[1, 1] - np.round(dim[1, 1]))
            >= 2 * np.prod(rho_dims) * eps
        ):
            val_error = """
                InvalidDim: The value of `dim` must evenly divide the number of
                rows and columns of `rho`; please provide the `dim` array
                containing the dimensions of the subsystems.
            """
            raise ValueError(val_error)

        dim[0, 1] = np.round(dim[0, 1])
        dim[1, 1] = np.round(dim[1, 1])
        num_sys = 2
    else:
        num_sys = len(dim)

    # Verify that the input sys makes sense.
    if any(sys) < 1 or any(sys) > num_sys:
        val_error = """
            InvalidSys: The subsystems in `sys` must be between 1 and 
            `len(dim).` inclusive.
        """
        raise ValueError(val_error)
    if len(sys) != 2:
        val_error = """
            InvalidSys: `sys` must be a vector with exactly two elements.
        """
        raise ValueError(val_error)

    # Swap the indicated subsystems.
    perm = list(range(1, num_sys + 1))
    perm = perm[::-1]
    return permute_systems(rho, perm, dim, row_only)


def swap_operator(dim: Union[List[int], int], is_sparse: bool = False) -> np.ndarray:
    r"""
    Produce a unitary operator that swaps two subsystems.

    Provides the unitary operator that swaps two copies of `dim`-dimensional
    space. If the two subsystems are not of the same dimension, `dim` should
    be a 1-by-2 vector containing the dimension of the subsystems.

    Examples
    ==========

    The $2$-dimensional swap operator is given by the following matrix

    .. math::
        X_2 =
        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1
        \end{pmatrix}

    Using `toqito` we can obtain this matrix as follows.

    >>> from toqito.perms import swap_operator
    >>> swap_operator(2)
    [[1., 0., 0., 0.],
     [0., 0., 1., 0.],
     [0., 1., 0., 0.],
     [0., 0., 0., 1.]]

    The :math:`3-`dimensional operator may be obtained using `toqito` as
    follows.

    >>> from toqito.perms import swap_operator
    >>> swap_operator(3)
    [[1., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 1., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 1., 0., 0.],
     [0., 1., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 1., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 1., 0.],
     [0., 0., 1., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 1., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 1.]]

    :param dim: The dimensions of the subsystems.
    :param is_sparse: Sparse if `True` and non-sparse if `False`.
    :return: The swap operator of dimension `dim`.
    """
    # Allow the user to enter a single number for dimension.
    if isinstance(dim, int):
        dim = np.array([dim, dim])

    # Swap the rows of the identity appropriately.
    return swap(iden(int(np.prod(dim)), is_sparse), [1, 2], dim, True)


def symmetric_projection(
    dim: int, p_val: int = 2, partial: bool = False
) -> [np.ndarray, sparse.lil_matrix]:
    r"""
    Produce the projection onto the symmetric subspace.

    Produces the orthogonal projection onto the symmetric subspace of `p`
    copies of `dim`-dimensional space. If `partial = True`, then the symmetric
    projection (PS) isn't the orthogonal projection itself, but rather a matrix
    whose columns form an orthonormal basis for the symmetric subspace (and
    hence the PS * PS' is the orthogonal projection onto the symmetric
    subspace.)

    Examples
    ==========

    The :math:`2`-dimensional symmetric projection with :math:`p=1` is given as
    :math:`2`-by-:math:`2` identity matrix

    .. math::
        \begin{pmatrix}
            1 & 0 \\
            0 & 1
        \end{pmatrix}.

    Using `toqito`, we can see this gives the proper result.

    >>> from toqito.perms import symmetric_projection
    >>> symmetric_projection(2, 1).todense()
    [[1., 0.],
     [0., 1.]]

    When :math:`d = 2` and :math:`p = 2` we have that

    .. math::
        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1/2 & 1/2 & 0 \\
            0 & 1/2 & 1/2 & 0 \\
            0 & 0 & 0 & 1
        \end{pmatrix}.

    Using `toqito` we can see this gives the proper result.

    >>> from toqito.perms import symmetric_projection
    >>> symmetric_projection(dim=2).todense()
    [[1. , 0. , 0. , 0. ],
     [0. , 0.5, 0.5, 0. ],
     [0. , 0.5, 0.5, 0. ],
     [0. , 0. , 0. , 1. ]]

    :param dim: The dimension of the local systems.
    :param p_val: Default value of 2.
    :param partial: Default value of 0.
    :return: Projection onto the symmetric subspace.
    """
    dimp = dim ** p_val

    if p_val == 1:
        return sparse.eye(dim)

    p_list = np.array(list(permutations(np.arange(1, p_val + 1))))
    p_fac = np.math.factorial(p_val)
    sym_proj = sparse.lil_matrix((dimp, dimp))

    for j in range(p_fac):
        sym_proj += permutation_operator(
            dim * np.ones(p_val), p_list[j, :], False, True
        )
    sym_proj = sym_proj / p_fac

    if partial:
        sym_proj = sym_proj.todense()
        sym_proj = sparse.lil_matrix(linalg.orth(sym_proj))
    return sym_proj


@dataclass
class UniqueElement:
    """Class for unique elements to keep track of occurrences."""

    value: int
    occurrences: int


def unique_perms(elements: List[int]):
    r"""
    Determine the number of unique permutations of a list.

    Examples
    ==========

    Consider the following vector

    .. math::
        \left[1, 1, 2, 2, 1, 2, 1, 3, 3, 3\right].

    The number of possible permutations possible with the above vector is
    :math:`4200`. This can be obtained using the `toqito` package as follows.

    >>> from toqito.perms import unique_perms
    >>> vec = [1, 1, 2, 2, 1, 2, 1, 3, 3, 3]
    >>> len(list(unique_perms(vec)))
    4200

    :param elements: List of integers.
    :return: The number of possible permutations possible.
    """
    elem_set = set(elements)
    list_unique = [
        UniqueElement(value=i, occurrences=elements.count(i)) for i in elem_set
    ]
    len_elems = len(elements)

    return perm_unique_helper(list_unique, [0] * len_elems, len_elems - 1)


def perm_unique_helper(
    list_unique: List[UniqueElement], result_list: List[int], elem_d: int
):
    """
    Provide helper function for unique_perms.

    :param list_unique:
    :param result_list:
    :param elem_d:
    :return:
    """
    if elem_d < 0:
        yield tuple(result_list)
    else:
        for i in list_unique:
            if i.occurrences > 0:
                result_list[elem_d] = i.value
                i.occurrences -= 1
                for g_perm in perm_unique_helper(list_unique, result_list, elem_d - 1):
                    yield g_perm
                i.occurrences += 1
