"""Check if matrix is nonnegative."""

import numpy as np
from toqito.matrix_props import is_positive_semidefinite

def is_nonnegative(input_mat: np.ndarray, mat_type: str = "nonnegative") -> bool:
    r"""Check if the matrix is nonnegative.

    When all the entries in the matrix are larger than or equal to zero the matrix of interest is a
    nonnegative matrix :cite:`WikiNonNegative`.


    Examples
    ==========
    We expect an identity matrix to be nonnegative.

    >>> import numpy as np
    >>> from toqito.matrix_props import is_nonnegative
    >>> is_nonnegative(np.identity(3))
    True

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    input_mat: np.ndarray
        Matrix of interest.

    """
    if mat_type == "nonnegative":
        if np.all(input_mat >= 0):
            return True
        else:
            return False
    elif mat_type == "doubly":
        if np.all(input_mat >= 0) and is_positive_semidefinite(input_mat):
            return True
        else:
            return False
    else:
        raise TypeError("Invalid matrix check type provided.")

