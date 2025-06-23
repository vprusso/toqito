"""Generates the realignment channel of a matrix."""

import numpy as np

from toqito.matrix_ops import partial_transpose
from toqito.perms import swap


def realignment(input_mat: np.ndarray, dim: int | list[int] = None) -> np.ndarray:
    r"""Compute the realignment of a bipartite operator :footcite:`Lupo_2008_Bipartite`.

    Gives the realignment of the matrix :code:`input_mat`, where it is assumed that the number
    of rows and columns of :code:`input_mat` are both perfect squares and both subsystems have
    equal dimension. The realignment is defined by mapping the operator :math:`|ij \rangle
    \langle kl |` to :math:`|ik \rangle \langle jl |` and extending linearly.

    If :code:`input_mat` is non-square, different row and column dimensions can be specified by
    putting the row dimensions in the first row of :code:`dim` and the column dimensions in the
    second row of :code:`dim`.

    Examples
    ==========

    The standard realignment map

    Using :code:`|toqito⟩`, we can generate the standard realignment map as follows. When viewed as a
    map on block matrices, the realignment map takes each block of the original matrix and makes
    its vectorization the rows of the realignment matrix. This is illustrated by the following
    small example:

    .. jupyter-execute::

     import numpy as np
     from toqito.channels import realignment

     test_input_mat = np.arange(1, 17).reshape(4, 4)

     realignment(test_input_mat)

    References
    ==========
    .. footbibliography::



    :param input_mat: The input matrix.
    :param dim: Default has all equal dimensions.
    :raises ValueError: If dimension of matrix is invalid.
    :return: The realignment map matrix.

    """
    dim_mat = input_mat.shape
    round_dim = np.round(np.sqrt(dim_mat))
    if dim is None:
        dim = np.transpose(np.array([round_dim]))
    if isinstance(dim, list):
        dim = np.array(dim)

    if isinstance(dim, int):
        dim = np.array([int(dim), int(dim_mat[0] / dim)])
        dim[1] = np.round(dim[1])

    # Dimension if row vector.
    if len(dim.shape) == 1:
        dim = dim[:].T
        dim = np.array([dim, dim])

    # Dimension is column vector.
    if min(dim.shape) == 1:
        dim = dim[:].T[0]
        dim = np.array([dim, dim])

    dim_x = np.array([[dim[0][1], dim[0][0]], [dim[1][0], dim[1][1]]])
    dim_x = np.int_(dim_x)
    dim_y = np.array([[dim[1][0], dim[0][0]], [dim[0][1], dim[1][1]]])

    x_tmp = swap(input_mat, [1, 2], dim, True)
    y_tmp = partial_transpose(x_tmp, [0], dim_x)

    return swap(y_tmp, [1, 2], dim_y, True)
