"""Checks if the matrix is nonnegative or doubly nonnegative."""

import numpy as np

from toqito.matrix_props import is_positive_semidefinite


def is_nonnegative(input_mat: np.ndarray, mat_type: str = "nonnegative") -> bool:
    r"""Check if the matrix is nonnegative.

    When all the entries in the matrix are larger than or equal to zero the matrix of interest is a
    nonnegative matrix :footcite:`WikiNonNegative`.

    When a matrix is nonegative and positive semidefinite :footcite:`WikiPosDef`, the matrix is doubly nonnegative.


    Examples
    ==========
    We expect an identity matrix to be nonnegative.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import is_nonnegative

     print(is_nonnegative(np.eye(2)))
     print(is_nonnegative(np.eye(2), "doubly"))
     print(is_nonnegative(np.array([[1, -1], [1, 1]])))


    References
    ==========
    .. footbibliography::


    :param input_mat: np.ndarray
                    Matrix of interest.
    :param mat_type: Type of nonnegative matrix. :code:`"nonnegative"` for a nonnegative matrix and :code:`"doubly"`
                    for a doubly nonnegative matrix.
    :raises TypeError: If something other than :code:`"doubly"`or :code:`"nonnegative"` is used for :code:`mat_type`.


    """
    valid_types = {"nonnegative", "doubly"}
    if mat_type not in valid_types:
        raise TypeError(f"Invalid matrix check type: {mat_type}. Must be one of: {valid_types}.")

    is_entrywise_nonnegative = bool(np.all(input_mat >= 0))

    if mat_type == "doubly":
        return is_entrywise_nonnegative and is_positive_semidefinite(input_mat)
    else:
        return is_entrywise_nonnegative
