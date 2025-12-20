"""Generates the realignment channel of a matrix."""

import numpy as np
from toqito.matrix_ops import partial_transpose
from toqito.perms import swap


def realignment(input_mat: np.ndarray, dim: int | list[int] = None) -> np.ndarray:
    r"""Compute the realignment of a bipartite operator :footcite:`Lupo_2008_Bipartite`.

    Gives the realignment of the matrix :code:`input_mat`, where it is assumed that
    the number of rows and columns of :code:`input_mat` are both perfect squares
    and both subsystems have equal dimension.

    :raises ValueError: If dimension of matrix is invalid.
    """

    # ============================
    # ✅ ADDED VALIDATION (ONLY)
    # ============================

    if not isinstance(input_mat, np.ndarray):
        raise ValueError("input_mat must be a NumPy ndarray.")

    if input_mat.ndim != 2:
        raise ValueError("input_mat must be a 2-dimensional matrix.")

    dim_mat = input_mat.shape

    if dim is None:
        sqrt_dims = np.sqrt(dim_mat)
        if not np.allclose(sqrt_dims, np.round(sqrt_dims)):
            raise ValueError(
                "input_mat dimensions must be perfect squares when dim is None."
            )

    if dim is not None:
        dim_arr = np.asarray(dim)

        if np.any(dim_arr <= 0):
            raise ValueError("All entries in dim must be positive.")

        if dim_arr.ndim > 2:
            raise ValueError("dim must be an int, 1D list, or 2D list.")

        # If dim is 1D, assume square bipartite system
        if dim_arr.ndim == 1:
            if np.prod(dim_arr) != dim_mat[0] or np.prod(dim_arr) != dim_mat[1]:
                raise ValueError(
                    "Provided dim does not match input_mat dimensions."
                )

        # If dim is 2D, check row/column consistency
        if dim_arr.ndim == 2:
            if (
                np.prod(dim_arr[0]) != dim_mat[0]
                or np.prod(dim_arr[1]) != dim_mat[1]
            ):
                raise ValueError(
                    "Provided dim does not match input_mat dimensions."
                )


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

