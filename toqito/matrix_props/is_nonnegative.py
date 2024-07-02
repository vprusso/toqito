"""Check if matrix is nonnegative."""

import numpy as np


def is_nonnegative(input_mat: np.ndarray) -> bool:
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
    return np.all(input_mat >= 0)
