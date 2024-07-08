"""Check if matrix is doubly nonnegative."""

import numpy as np

from toqito.matrix_props import is_nonnegative, is_positive_semidefinite


def is_doubly_nonnegative(input_mat: np.ndarray) -> bool:
    r"""Check if the matrix is doubly nonnegative.

    When a matrix is nonnegative :cite:`WikiNonNegative` and is positive semidefinite :cite:`WikiPosDef`.


    Examples
    ==========
    We expect an identity matrix to be doubly nonnegative.

    >>> import numpy as np
    >>> from toqito.matrix_props import is_doubly_nonnegative
    >>> is_doubly_nonnegative(np.identity(3))
    True

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    input_mat: np.ndarray
        Matrix of interest.

    """
    return is_nonnegative(input_mat) and is_positive_semidefinite(input_mat)
