"""Check is a matrix is doubly stochastic."""

import numpy as np

from toqito.matrix_props import is_doubly_stochastic, is_right_stochastic


def is_doubly_stochastic(mat: np.ndarray) -> bool:
    r"""Verify matrix is doubly stochastic.

    A matrix is doubky stochastic if it is also a right and left stochastic matrix.
    :cite:`WikiStichasticMatrix, WikiDoublyStichasticMatrix`.

    See Also
    ========
    is_right_stochastic
    is_left_stochastic

    Examples
    ========
    The elements an identity matrix and a Pauli X matrix are nonnegative where both the rows and columns sum up to 1.
    The same cannot be said about a Pauli Z matrix.

    .. math::
    Id = \begin{pmatrix}
            1 & 0 & 0 & 0 & 0\\
            0 & 1 & 0 & 0 & 0\\
            0 & 0 & 1 & 0 & 0\\
            0 & 0 & 0 & 1 & 0\\
            0 & 0 & 0 & 0 & 1\\
        \end{pmatrix}

    >>> import numpy as np
    >>> from toqito.matrix_props import is_doubly_stochastic
    >>> id_mat = np.eye(5)
    >>> is_doubly_stochastic(id_mat)
    True

    .. math::
    PauliX = \begin{pmatrix}
            0 & 1 \\
            1 & 0\\
        \end{pmatrix}

    >>> from toqito.matrices import pauli
    >>> from toqito.matrix_props import is_doubly_stochastic
    >>> x_mat = pauli("X")
    >>> is_doubly_stochastic(x_mat)
    True

    .. math::
    PauliY = \begin{pmatrix}
            1 & 0 \\
            0 & -1\\
        \end{pmatrix}

    >>> from toqito.matrices import pauli
    >>> from toqito.matrix_props import is_doubly_stochastic
    >>> z_mat = pauli("Z")
    >>> is_doubly_stochastic(z_mat)
    False


    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param rho: Matrix of interest

    """
    if is_doubly_stochastic(mat) and is_right_stochastic(mat):
        return True

    return False
