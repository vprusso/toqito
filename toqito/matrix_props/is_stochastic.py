"""Checks if the matrix is stochastic."""

import numpy as np

from toqito.matrix_props import is_nonnegative, is_square


def is_stochastic(mat: np.ndarray, mat_type: str) -> bool:
    r"""Verify matrix is doubly, right or left stochastic.

    When the nonnegative elements in a row of a square matrix sum up to 1, the matrix is right stochastic and if the
    columns sum up to 1, the matrix is left stochastic :footcite:`WikiStochasticMatrix`.

    When a matrix is right and left stochastic, it is a doubly stochastic matrix :footcite:`WikiDoublyStochasticMatrix`
    .

    See Also
    ========
    :py:func:`~toqito.matrix_props.is_doubly_stochastic.is_doubly_stochastic`

    Examples
    ========
    The elements of an identity matrix and a Pauli-X matrix are nonnegative such that the rows and columns sum up to 1.
    We expect these matrices to be left and right stochastic. The same cannot be said about a Pauli-Z or a Pauli-Y
    matrix.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_props import is_stochastic

     print(is_stochastic(np.eye(5), "right"))
     print(is_stochastic(np.eye(5), "left"))
     print(is_stochastic(np.eye(5), "doubly"))

    .. jupyter-execute::

     from toqito.matrices import pauli
     from toqito.matrix_props import is_stochastic

     print(is_stochastic(pauli("X"), "left"))
     print(is_stochastic(pauli("X"), "right"))
     print(is_stochastic(pauli("X"), "doubly"))

    .. jupyter-execute::

     from toqito.matrices import pauli
     from toqito.matrix_props import is_stochastic

     print(is_stochastic(pauli("Z"), "right"))
     print(is_stochastic(pauli("Z"), "left"))
     print(is_stochastic(pauli("Z"), "doubly"))

    References
    ==========
    .. footbibliography::


    :param mat: Matrix of interest
    :param mat_type: Type of stochastic matrix.
                   :code:`"left"` for left stochastic matrix and :code:`"right"` for right stochastic matrix
                   and :code:`"doubly"` for a doubly stochastic matrix.
    :return: Returns :code:`True` if the matrix is doubly, right or left stochastic, :code:`False` otherwise.
    :raises TypeError: If something other than :code:`"doubly"`, :code:`"left"` or :code:`"right"` is used for
                      :code:`mat_type`

    """
    if mat_type not in {"left", "right", "doubly"}:
        raise TypeError("Allowed stochastic matrix types are: left, right, and doubly.")

    if not (is_square(mat) and is_nonnegative(mat)):
        return False

    checks = []

    if mat_type in {"left", "doubly"}:
        col_sums = np.sum(mat, axis=0)
        checks.append(np.allclose(col_sums, 1.0))

    if mat_type in {"right", "doubly"}:
        row_sums = np.sum(mat, axis=1)
        checks.append(np.allclose(row_sums, 1.0))

    return all(checks)
