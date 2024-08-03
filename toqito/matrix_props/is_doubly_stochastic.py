"""Check is a matrix is doubly stochastic."""

import numpy as np

from toqito.matrix_props import is_stochastic


def is_doubly_stochastic(mat: np.ndarray) -> bool:
    r"""Verify matrix is doubly stochastic.

    A matrix is doubly stochastic if it is also a right and left stochastic matrix.
    :cite:`WikiStochasticMatrix, WikiDoublyStochasticMatrix`.

    See Also
    ========
    is_stochastic

    Examples
    ========
    The elements of an identity matrix and a Pauli-X matrix are nonnegative where both the rows and columns sum up to 1.
    The same cannot be said about a Pauli-Z matrix.

    >>> import numpy as np
    >>> from toqito.matrix_props import is_doubly_stochastic
    >>> is_doubly_stochastic(np.eye(5))
    True

    >>> from toqito.matrices import pauli
    >>> from toqito.matrix_props import is_doubly_stochastic
    >>> is_doubly_stochastic(pauli("X"))
    True


    >>> from toqito.matrices import pauli
    >>> from toqito.matrix_props import is_doubly_stochastic
    >>> is_doubly_stochastic(pauli("Z"))
    False


    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param rho: Matrix of interest

    """
    if is_stochastic(mat, "right") and is_stochastic(mat, "left"):
            return True

    return False
