"""Swap."""
from typing import List, Union

import numpy as np

from toqito.perms import permute_systems


def swap(
    rho: np.ndarray,
    sys: List[int] = None,
    dim: Union[List[int], List[List[int]], int, np.ndarray] = None,
    row_only: bool = False,
) -> np.ndarray:
    r"""
    Swap two subsystems within a state or operator.

    Swaps the two subsystems of the vector or matrix :code:`rho`, where the dimensions of the
    (possibly more than 2) subsystems are given by :code:`dim` and the indices of the two subsystems
    to be swapped are specified in the 1-by-2 vector :code:`sys`.

    If :code:`rho` is non-square and not a vector, different row and column dimensions can be
    specified by putting the row dimensions in the first row of :code:`dim` and the column
    dimensions in the second row of :code:`dim`.

    If :code:`row_only` is set to :code:`True`, then only the rows of :code:`rho` are swapped, but
    not the columns -- this is equivalent to multiplying :code:`rho` on the left by the
    corresponding swap operator, but not on the right.

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
        \end{pmatrix}.

    If we apply the :code:`swap` function provided by :code:`toqito` on :math:`X`, we should obtain
    the following matrix

    .. math::
        \text{Swap}(X) =
        \begin{pmatrix}
            1 & 9 & 5 & 13 \\
            3 & 11 & 7 & 15 \\
            2 & 10 & 6 & 14 \\
            4 & 12 & 8 & 16
        \end{pmatrix}.

    This can be observed by the following example in :code:`toqito`.

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

    It is also possible to use the :code:`sys` and :code:`dim` arguments, it is possible to specify
    the system and dimension on which to apply the swap operator. For instance for
    :code:`sys = [1 ,2]` and :code:`dim = 2` we have that

    .. math::
        Swap(X)_{2, [1, 2]} =
        \begin{pmatrix}
            1 & 9 & 5 & 13 \\
            3 & 11 & 7 & 15 \\
            2 & 10 & 6 & 14 \\
            4 & 12 & 8 & 16
        \end{pmatrix}.

    Using :code:`toqito` we can see this gives the proper result.

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

    It is also possible to perform the :code:`swap` function on vectors in addition to matrices.

    >>> from toqito.perms import swap
    >>> import numpy as np
    >>> test_vec = np.array([1, 2, 3, 4])
    >>> swap(test_vec)
    [1 3 2 4]

    :param rho: A vector or matrix to have its subsystems swapped.
    :param sys: Default: [1, 2]
    :param dim: Default: :code:`[sqrt(len(X), sqrt(len(X)))]`
    :param row_only: Default: :code:False
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
            np.abs(dim[0, 1] - np.round(dim[0, 1])) + np.abs(dim[1, 1] - np.round(dim[1, 1]))
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
    perm[sys[0] - 1 :] = perm[sys[0] - 1 :][::-1]
    return permute_systems(rho, perm, dim, row_only)
