"""Swap is used to apply the swap function within a quantum state or an operator."""

import numpy as np

from toqito.perms import permute_systems


def swap(
    rho: np.ndarray,
    sys: list[int] | None = None,
    dim: list[int] | list[list[int]] | int | np.ndarray | None = None,
    row_only: bool = False,
) -> np.ndarray:
    r"""Swap two subsystems within a state or operator.

    Swaps the two subsystems of the vector or matrix :code:`rho`, where the dimensions of the (possibly more than 2)
    subsystems are given by :code:`dim` and the indices of the two subsystems to be swapped are specified in the 1-by-2
    vector :code:`sys`.

    If :code:`rho` is non-square and not a vector, different row and column dimensions can be specified by putting the
    row dimensions in the first row of :code:`dim` and the column dimensions in the second row of :code:`dim`.

    If :code:`row_only` is set to :code:`True`, then only the rows of :code:`rho` are swapped, but not the columns --
    this is equivalent to multiplying :code:`rho` on the left by the corresponding swap operator, but not on the right.

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

    If we apply the :code:`swap` function provided by :code:`|toqito⟩` on :math:`X`, we should obtain the following
    matrix

    .. math::
        \text{Swap}(X) =
        \begin{pmatrix}
            1 & 9 & 5 & 13 \\
            3 & 11 & 7 & 15 \\
            2 & 10 & 6 & 14 \\
            4 & 12 & 8 & 16
        \end{pmatrix}.

    This can be observed by the following example in :code:`|toqito⟩`.

    .. jupyter-execute::

     import numpy as np
     from toqito.perms import swap

     test_mat = np.arange(1, 17).reshape(4, 4)

     swap(test_mat)

    It is also possible to use the :code:`sys` and :code:`dim` arguments, it is possible to specify the system and
    dimension on which to apply the swap operator. For instance for :code:`sys = [1 ,2]` and :code:`dim = 2` we have
    that

    .. math::
        \text{Swap}(X)_{2, [1, 2]} =
        \begin{pmatrix}
            1 & 9 & 5 & 13 \\
            3 & 11 & 7 & 15 \\
            2 & 10 & 6 & 14 \\
            4 & 12 & 8 & 16
        \end{pmatrix}.

    Using :code:`|toqito⟩` we can see this gives the proper result.

    .. jupyter-execute::

     import numpy as np
     from toqito.perms import swap

     test_mat = np.array(
         [[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]]
     )
     swap(test_mat, [1, 2], 2)

    It is also possible to perform the :code:`swap` function on vectors in addition to matrices.

    .. jupyter-execute::

     import numpy as np
     from toqito.perms import swap

     test_vec = np.array([1, 2, 3, 4])

     swap(test_vec)



    :raises ValueError: If dimension does not match the number of subsystems.
    :param rho: A vector or matrix to have its subsystems swapped.
    :param sys: Default: [1, 2]
    :param dim: Default: :code:`[sqrt(len(X), sqrt(len(X)))]`
    :param row_only: Default: :code:`False`
    :return: The swapped matrix.

    """
    if dim is not None and not isinstance(dim, (int, list, np.ndarray)):
        raise TypeError("dim must be None, int, list, or np.ndarray.")

    if len(rho.shape) == 1:
        rho_dims = (1, rho.shape[0])
    else:
        rho_dims = rho.shape

    round_dim = np.rint(np.sqrt(rho_dims)).astype(int)

    if sys is None:
        sys = [1, 2]

    if dim is None:
        # Assume square subsystems inferred from rho_dims.
        dim = np.array([[round_dim[0], round_dim[0]], [round_dim[1], round_dim[1]]], dtype=int)
        num_sys = len(dim)
    elif isinstance(dim, int):
        # Split dimensions into two factors: dim and rho_dim/dim.
        if rho_dims[0] % dim != 0 or rho_dims[1] % dim != 0:
            raise ValueError("InvalidDim: The value of dim must evenly divide the number of rows and columns of rho.")
        dim = np.array([[dim, rho_dims[0] // dim], [dim, rho_dims[1] // dim]], dtype=int)
        num_sys = 2
    elif isinstance(dim, (list, np.ndarray)):
        if not all(isinstance(d, (int, float, np.integer, np.floating)) for d in np.ravel(dim)):
            raise TypeError("dim entries must be int or float values.")
        dim = np.array(dim, dtype=int)
        num_sys = len(dim)

    if len(sys) != 2:
        raise ValueError("InvalidSys: sys must be a vector with exactly two elements.")

    if not (1 <= sys[0] <= num_sys and 1 <= sys[1] <= num_sys):
        raise ValueError("InvalidSys: The subsystems in sys must be between 1 and len(dim). inclusive.")

    # Swap the indicated subsystems.
    perm = np.arange(num_sys)
    sys = np.array(sys) - 1

    perm[sys] = perm[sys[::-1]]

    return permute_systems(input_mat=rho, perm=perm, dim=dim, row_only=row_only)
