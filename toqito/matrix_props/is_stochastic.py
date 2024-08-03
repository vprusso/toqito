"""Check is a matrix is stochastic."""

import numpy as np

from toqito.matrix_props import is_nonnegative, is_square


def is_stochastic(mat: np.ndarray, mat_type: str) -> bool:
    r"""Verify matrix is right or left stochastic.

    When the nonnegative elements in a row of a square matrix sum up to 1, the matrix is right stochastic and if the
    columns sum up to 1, the matrix is left stochastic. :cite:WikiStochasticMatrix.

    See Also
    ========
    is_doubly_stochastic

    Examples
    ========
    The elements of an identity matrix and a Pauli-X matrix are nonnegative and the rows sum up to 1. The same cannot be
    said about a Pauli-Z or a Pauli-Y matrix.

     >>> import numpy as np
     >>> from toqito.matrix_props import is_stochastic
     >>> is_stochastic(np.eye(5), "right")
     True
     >>> is_stochastic(np.eye(5), "left")
     True

     >>> from toqito.matrices import pauli
     >>> from toqito.matrix_props import is_stochastic
     >>> is_stochastic(pauli("X"), "left")
     True
     >>> is_stochastic(pauli("X"), "right")
     True

     >>> from toqito.matrices import pauli
     >>> from toqito.matrix_props import is_stochastic
     >>> is_stochastic(pauli("Z"), "right")
     False
     >>> is_stochastic(pauli("Z"), "left")
     False




    References
    ==========
    .. bibliography::
          :filter: docname in docnames

     :param rho: Matrix of interest
     :param mat_type: Type of stochastic matrix.
          "left" for left stochastic matrix and "right" for right stochastic matrix.

    """
    if mat_type == "left":
     if is_square(mat) and is_nonnegative(mat) and np.all(np.sum(mat, axis=0) == 1.0):
        return True

     return False

    elif mat_type == "right":
     if is_square(mat) and is_nonnegative(mat) and np.all(np.sum(mat, axis=1) == 1.0):
        return True

     return False
    else:
      raise TypeError("Invalid stochastic matrix type provided.")

