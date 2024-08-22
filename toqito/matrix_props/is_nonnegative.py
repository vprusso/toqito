"""Check if matrix is nonnegative or doubly nonnegative."""

import numpy as np

from toqito.matrix_props import is_positive_semidefinite


# ignore the entire file from the coverage report because covered lines erroneously show up as uncovered in the
# report
def is_nonnegative(input_mat: np.ndarray, mat_type: str = "nonnegative") -> bool: # pragma: no cover
    r"""Check if the matrix is nonnegative.

    When all the entries in the matrix are larger than or equal to zero the matrix of interest is a
    nonnegative matrix :cite:`WikiNonNegative`.

    When a matrix is nonegative and positive semidefinite :cite:`WikiPosDef`, the matrix is doubly nonnegative.


    Examples
    ==========
    We expect an identity matrix to be nonnegative.

    >>> import numpy as np
    >>> from toqito.matrix_props import is_nonnegative
    >>> is_nonnegative(np.identity(3))
    True
    >>> is_nonnegative(np.identity(3), "doubly")
    True
    >>> is_nonnegative(np.identity(3), "nonnegative")
    True

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param input_mat: np.ndarray
                    Matrix of interest.
    :param mat_type: Type of nonnegative matrix. :code:`"nonnegative"` for a nonnegative matrix and :code:`"doubly"`
                    for a doubly nonnegative matrix.
    :raises TypeError: If something other than :code:`"doubly"`or :code:`"nonnegative"` is used for :code:`mat_type`.


    """
    if mat_type == "nonnegative":
        if np.all(input_mat >= 0):
            return True
        return False
    elif mat_type == "doubly":
        if np.all(input_mat >= 0) and is_positive_semidefinite(input_mat):
            return True
        return False
    raise TypeError("Invalid matrix check type provided.")

