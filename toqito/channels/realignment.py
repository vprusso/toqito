"""Generates the realignment channel of a matrix."""

import numpy as np

from toqito.matrix_ops import partial_transpose
from toqito.perms import swap


def realignment(input_mat: np.ndarray, dim: int | list[int] | np.ndarray | None = None) -> np.ndarray:
    r"""Compute the realignment of a bipartite operator [@lupo2008bipartite].

    Gives the realignment of the matrix `input_mat`, where it is assumed that the number
    of rows and columns of `input_mat` are both perfect squares and both subsystems have
    equal dimension. The realignment is defined by mapping the operator \(|ij \rangle
    \langle kl |\) to \(|ik \rangle \langle jl |\) and extending linearly.

    If `input_mat` is non-square, different row and column dimensions can be specified by
    putting the row dimensions in the first row of `dim` and the column dimensions in the
    second row of `dim`.

    Examples:
        The standard realignment map

        Using `|toqito⟩`, we can generate the standard realignment map as follows. When viewed as a
        map on block matrices, the realignment map takes each block of the original matrix and makes
        its vectorization the rows of the realignment matrix. This is illustrated by the following
        small example:

        ```python exec="1" source="above"
        import numpy as np
        from toqito.channels import realignment

        test_input_mat = np.arange(1, 17).reshape(4, 4)

        print(realignment(test_input_mat))
        ```

    Raises:
        ValueError: If dimension of matrix is invalid.

    Args:
        input_mat: The input matrix.
        dim: Default has all equal dimensions.

    Returns:
        The realignment map matrix.

    """
    dim_mat = input_mat.shape
    round_dim = np.round(np.sqrt(dim_mat))
    if dim is None:
        dim_arr = np.transpose(np.array([round_dim]))
    elif isinstance(dim, list):
        dim_arr = np.array(dim)
    elif isinstance(dim, int):
        dim_arr = np.array([int(dim), int(dim_mat[0] / dim)])
        dim_arr[1] = np.round(dim_arr[1])
    else:
        dim_arr = dim

    # Dimension if row vector.
    if len(dim_arr.shape) == 1:
        dim_arr = dim_arr[:].T
        dim_arr = np.array([dim_arr, dim_arr])

    # Dimension is column vector.
    if min(dim_arr.shape) == 1:
        dim_arr = dim_arr[:].T[0]
        dim_arr = np.array([dim_arr, dim_arr])

    dim_x = np.array([[dim_arr[0][1], dim_arr[0][0]], [dim_arr[1][0], dim_arr[1][1]]])
    dim_x = np.int_(dim_x)
    dim_y = np.array([[dim_arr[1][0], dim_arr[0][0]], [dim_arr[0][1], dim_arr[1][1]]])

    x_tmp = swap(input_mat, [1, 2], dim_arr, True)
    y_tmp = partial_transpose(x_tmp, [0], dim_x)

    return swap(y_tmp, [1, 2], dim_y, True)
