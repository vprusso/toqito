"""The partial transpose."""
from typing import List, Union

import numpy as np

from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from toqito.perms import permute_systems
from toqito.helper import expr_as_np_array, np_array_as_expr


def partial_transpose(
    rho: Union[np.ndarray, Variable],
    sys: Union[List[int], np.ndarray, int] = 2,
    dim: Union[List[int], np.ndarray] = None,
) -> Union[np.ndarray, Expression]:
    r"""Compute the partial transpose of a matrix [WikPtrans]_.

    The *partial transpose* is defined as

    .. math::
        \left( \text{T} \otimes \mathbb{I}_{\mathcal{Y}} \right)
        \left(X\right)

    where :math:`X \in \text{L}(\mathcal{X})` is a linear operator over the complex Euclidean
    space :math:`\mathcal{X}` and where :math:`\text{T}` is the transpose mapping
    :math:`\text{T} \in \text{T}(\mathcal{X})` defined as

    .. math::
        \text{T}(X) = X^{\text{T}}

    for all :math:`X \in \text{L}(\mathcal{X})`.

    By default, the returned matrix is the partial transpose of the matrix :code:`rho`, where it
    is assumed that the number of rows and columns of :code:`rho` are both perfect squares and
    both subsystems have equal dimension. The transpose is applied to the second subsystem.

    In the case where :code:`sys` amd :code:`dim` are specified, this function gives the partial
    transpose of the matrix :code:`rho` where the dimensions of the (possibly more than 2)
    subsystems are given by the vector :code:`dim` and the subsystems to take the partial
    transpose are given by the scalar or vector :code:`sys`. If :code:`rho` is non-square,
    different row and column dimensions can be specified by putting the row dimensions in the
    first row of :code:`dim` and the column dimensions in the second row of :code:`dim`.

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

    By default, in :code:`toqito`, the partial transpose function performs the transposition on
    the second subsystem as follows.

    >>> from toqito.channels import partial_transpose
    >>> import numpy as np
    >>> test_input_mat = np.arange(1, 17).reshape(4, 4)
    >>> partial_transpose(test_input_mat)
    [[ 1  5  3  7]
     [ 2  6  4  8]
     [ 9 13 11 15]
     [10 14 12 16]]

    By specifying the :code:`sys = 1` argument, we can perform the partial transpose over the
    first subsystem (instead of the default second subsystem as done above). Performing the
    partial transpose over the first subsystem yields the following matrix

    .. math::
        X_{pt, 1} = \begin{pmatrix}
                        1 & 2 & 9 & 10 \\
                        5 & 6 & 13 & 14 \\
                        3 & 4 & 11 & 12 \\
                        7 & 8 & 15 & 16
                    \end{pmatrix}.

    >>> from toqito.channels import partial_transpose
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
    .. [WikPtrans] Wikipedia: Partial transpose
        https://en.wikipedia.org/w/index.php?title=Partial_transpose

    :param rho: A matrix.
    :param sys: Scalar or vector specifying the size of the subsystems.
    :param dim: Dimension of the subsystems. If :code:`None`, all dimensions
                are assumed to be equal.
    :returns: The partial transpose of matrix :code:`rho`.
    """
    # If the input matrix is a CVX variable for an SDP, we convert it to a
    # numpy array, perform the partial transpose, and convert it back to a CVX
    # variable.
    if isinstance(rho, Variable):
        rho_np = expr_as_np_array(rho)
        transposed_rho = partial_transpose(rho_np, sys, dim)
        transposed_rho = np_array_as_expr(transposed_rho)
        return transposed_rho

    sqrt_rho_dims = np.round(np.sqrt(list(rho.shape)))

    if dim is None:
        dim = np.array([[sqrt_rho_dims[0], sqrt_rho_dims[0]], [sqrt_rho_dims[1], sqrt_rho_dims[1]]])
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
        if np.abs(dim[1] - np.round(dim[1]))[0] >= 2 * list(rho.shape)[0] * np.finfo(float).eps:
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
    z_tmp = np.reshape(
        y_tmp,
        [
            int(sub_sys_vec_r[0]) * int(sub_prod_c),
            int(sub_sys_vec_c[0]) * int(sub_prod_r),
        ],
        order="F",
    )

    # If z_tmp is a just a 1-D matrix, extract just the row.
    if z_tmp.shape[0] == 1:
        z_tmp = z_tmp[0]

    # Return the subsystems back to their original positions.
    dim[[0, 1], sys - 1] = dim[[1, 0], sys - 1]

    return permute_systems(z_tmp, perm, dim, False, True)
