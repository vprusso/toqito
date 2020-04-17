"""Computes the partial transpose of a matrix."""
from typing import List, Union
import numpy as np
from toqito.perms.permute_systems import permute_systems


def partial_transpose(
    rho: np.ndarray,
    sys: Union[List[int], np.ndarray, int] = 2,
    dim: Union[List[int], np.ndarray] = None,
) -> np.ndarray:
    r"""Compute the partial transpose of a matrix [4]_.

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

    Examples
    ==========

    Consider the following matrix

    .. math::
        X = \begin{pmatrix}
                1 & 2 & 3 & 4 \\
                5 & 6 & 7 & 8 \\
                9 & 10 & 11 & 12 \\
                13 & 14 & 15 & 16
            \end{pmatrix}.

    Performing the partial transpose on the matrix :math:`X` over the second
    subsystem yields the following matrix

    .. math::
        X_{pt, 2} = \begin{pmatrix}
                    1 & 5 & 3 & 7 \\
                    2 & 6 & 4 & 8 \\
                    9 & 13 & 11 & 15 \\
                    10 & 14 & 12 & 16
                 \end{pmatrix}.

    By default, in `toqito`, the partial transpose function performs the
    transposition on the second subsystem as follows.

    >>> from toqito.maps.partial_transpose import partial_transpose
    >>> import numpy as np
    >>> test_input_mat = np.arange(1, 17).reshape(4, 4)
    >>> partial_transpose(test_input_mat)
    [[ 1  5  3  7]
     [ 2  6  4  8]
     [ 9 13 11 15]
     [10 14 12 16]]

    By specifying the `sys = 1` argument, we can perform the partial transpose
    over the first subsystem (instead of the default second subsystem as done
    above). Performing the partial transpose over the first subsystem yields the
    following matrix

    .. math::
        X_{pt, 1} = \begin{pmatrix}
                        1 & 2 & 9 & 10 \\
                        5 & 6 & 13 & 14 \\
                        3 & 4 & 11 & 12 \\
                        7 & 8 & 15 & 16
                    \end{pmatrix}

    >>> from toqito.maps.partial_transpose import partial_transpose
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> )
    >>> partial_transpose(test_input_mat, 1)
    [[ 1  2  9 10]
     [ 5  6 13 14]
     [ 3  4 11 12]
     [ 7  8 15 16]]

    References
    ==========
    .. [4] Wikipedia: Partial transpose
        https://en.wikipedia.org/w/index.php?title=Partial_transpose&redirect=no

    :param rho: A matrix.
    :param sys: Scalar or vector specifying the size of the subsystems.
    :param dim: Dimension of the subsystems. If `None`, all dimensions
                are assumed to be equal.
    :returns: The partial transpose of matrix `rho`.
    """
    sqrt_rho_dims = np.round(np.sqrt(list(rho.shape)))

    if dim is None:
        dim = np.array(
            [[sqrt_rho_dims[0], sqrt_rho_dims[0]], [sqrt_rho_dims[1], sqrt_rho_dims[1]]]
        )
    if isinstance(dim, float):
        dim = np.array([dim])
    if isinstance(dim, list):
        dim = np.array(dim)
    if isinstance(sys, list):
        sys = np.array(sys)
    if isinstance(sys, int):
        sys = np.array([sys])

    num_sys = len(dim)
    # Allow the user to enter a single number for dim.
    if num_sys == 1:
        dim = np.array([dim, list(rho.shape)[0] / dim])
        if (
            np.abs(dim[1] - np.round(dim[1]))[0]
            >= 2 * list(rho.shape)[0] * np.finfo(float).eps
        ):
            raise ValueError(
                "InvalidDim: If `dim` is a scalar, `rho` must be "
                "square and `dim` must evenly divide `len(rho)`; "
                "please provide the `dim` array containing the "
                "dimensions of the subsystems."
            )
        dim[1] = np.round(dim[1])
        num_sys = 2

    # Allow the user to enter a vector for dim if X is square.
    if min(dim.shape) == 1 or len(dim.shape) == 1:
        # Force dim to be a row vector.
        dim = dim.T.flatten()
        dim = np.array([dim, dim])

    prod_dim_r = int(np.prod(dim[0, :]))
    prod_dim_c = int(np.prod(dim[1, :]))

    sub_prod_r = np.prod(dim[0, sys - 1])
    sub_prod_c = np.prod(dim[1, sys - 1])

    sub_sys_vec_r = prod_dim_r * np.ones(int(sub_prod_r)) / sub_prod_r
    sub_sys_vec_c = prod_dim_c * np.ones(int(sub_prod_c)) / sub_prod_c

    set_diff = list(set(list(range(1, num_sys + 1))) - set(sys))
    perm = sys.tolist()[:]
    perm.extend(set_diff)

    # Permute the subsystems so that we just have to do the partial transpose
    # on the first (potentially larger) subsystem.
    rho_permuted = permute_systems(rho, perm, dim)

    x_tmp = np.reshape(
        rho_permuted,
        [
            int(sub_sys_vec_r[0]),
            int(sub_prod_r),
            int(sub_sys_vec_c[0]),
            int(sub_prod_c),
        ],
        order="F",
    )
    y_tmp = np.transpose(x_tmp, [0, 3, 2, 1])
    z_tmp = np.reshape(y_tmp, [int(prod_dim_r), int(prod_dim_c)], order="F")

    # # Return the subsystems back to their original positions.
    # if len(sys) > 1:
    #     dim[:, sys-1] = dim[[1, 0], sys-1]

    dim = dim[:, np.array(perm) - 1]

    return permute_systems(z_tmp, perm, dim, False, True)
