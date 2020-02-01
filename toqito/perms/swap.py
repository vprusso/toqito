import numpy as np
from toqito.perms.permute_systems import permute_systems


def swap(input_mat: np.ndarray,
         sys=None,
         dim=None,
         row_only: bool = False) -> np.ndarray:
    """
    Swap two subsystems within a state or operator.

    Swaps the two subsystems of the vector or matrix X, where the dimensions
    of the (possibly more than 2) subsystems are given by DIM and the indices
    of the two subsystems to be swapped are specified in the 1-by-2 vector SYS.

    If X is non-square and not a vector, different row and column dimensions
    can be specified by putting the row dimensions in the first row of DIM and
    the column dimensions in the second row of DIM.

    If ROW_ONLY is set to True, then only the rows of X are swapped, but not
    the columns -- this is equivalent to multiplying X on the left by the
    corresponding swap operator, but not on the right.

    :param input_mat: A vector or matrix to have its subsystems swapped.
    :param sys: Default: [1, 2]
    :param dim: Default: [sqrt(len(X), sqrt(len(X)))]
    :param row_only: Default: False
    :return: The swapped matrix.
    """
    eps = np.finfo(float).eps
    if len(input_mat.shape) == 1:
        dX = (1, input_mat.shape[0])
    else:
        dX = input_mat.shape

    round_dim = np.round(np.sqrt(dX))

    if sys is None:
        sys = [1, 2]

    if isinstance(dim, list):
        dim = np.array(dim)
    if dim is None:
        dim = np.array([[round_dim[0], round_dim[0]],
                        [round_dim[1], round_dim[1]]])

    if isinstance(dim, int):
        dim = np.array([[dim, dX[0]/dim],
                        [dim, dX[1]/dim]])
        if np.abs(dim[0, 1] - np.round(dim[0, 1])) + \
           np.abs(dim[1, 1] - np.round(dim[1, 1])) >= 2 * np.prod(dX) * eps:
            val_error = """
                InvalidDim: The value of DIM must evenly divide the number of
                rows and columns of X; please provide the DIM array containing
                the dimensions of the subsystems.
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
            InvalidSys: The subsystems in SYS must be between 1 and len(DIM)
            inclusive.
        """
        raise ValueError(val_error)
    elif len(sys) != 2:
        val_error = """
            InvalidSys: SYS must be a vector with exactly two elements.
        """
        raise ValueError(val_error)

    # Swap the indicated subsystems.
    perm = list(range(1, num_sys+1))
    perm = perm[::-1]
    return permute_systems(input_mat, perm, dim, row_only)

