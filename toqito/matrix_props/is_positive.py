"""Checks if the matrix is positive."""

import numpy as np


def is_positive(input_mat: np.ndarray) -> bool:
    r"""Check if the matrix is positive.

    When all the entries in the matrix are larger than zero the matrix of interest is a
    positive matrix :footcite:`WikiNonNegative`.

    .. note::
        This function is different from :any:`matrix_props.is_positive_definite`,
        :any:`matrix_props.is_totally_positive` and :any:`matrix_props.is_positive_semidefinite`.


    Examples
    ==========
    We expect a matrix full of 1s to be positive.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import is_positive

     input_mat = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]])

     is_positive(input_mat)

    References
    ==========
    .. footbibliography::



    input_mat: np.ndarray
        Matrix of interest.

    """
    return bool(np.all(input_mat > 0))
