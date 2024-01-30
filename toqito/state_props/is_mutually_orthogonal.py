"""Check if states are mutually orthogonal."""


from typing import Any

import numpy as np


def is_mutually_orthogonal(vec_list: list[np.ndarray | list[float | Any]]) -> bool:
    r"""Check if list of vectors are mutually orthogonal :cite:`WikiOrthog`.

    We say that two bases

    .. math::
        \begin{equation}
            \mathcal{B}_0 = \left\{u_a: a \in \Sigma \right\} \subset \mathbb{C}^{\Sigma}
            \quad \text{and} \quad
            \mathcal{B}_1 = \left\{v_a: a \in \Sigma \right\} \subset \mathbb{C}^{\Sigma}
        \end{equation}

    are *mutually orthogonal* if and only if
    :math:`\left|\langle u_a, v_b \rangle\right| = 0` for all :math:`a, b \in \Sigma`.

    For :math:`n \in \mathbb{N}`, a set of bases :math:`\left\{
    \mathcal{B}_0, \ldots, \mathcal{B}_{n-1} \right\}` are mutually orthogonal if and only if
    every basis is orthogonal with every other basis in the set, i.e. :math:`\mathcal{B}_x`
    is orthogonal with :math:`\mathcal{B}_x^{\prime}` for all :math:`x \not= x^{\prime}` with
    :math:`x, x^{\prime} \in \Sigma`.

    Examples
    ==========

    The Bell states constitute a set of mutually orthogonal vectors.

    >>> from toqito.states import bell
    >>> from toqito.state_props import is_mutually_orthogonal
    >>> states = [bell(0), bell(1), bell(2), bell(3)]
    >>> is_mutually_orthogonal(states)
    True

    The following is an example of a list of vectors that are not mutually orthogonal.

    >>> import numpy as np
    >>> from toqito.states import bell
    >>> from toqito.state_props import is_mutually_orthogonal
    >>> states = [np.array([1, 0]), np.array([1, 1])]
    >>> is_mutually_orthogonal(states)
    False

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: If at least two vectors are not provided.
    :param vec_list: The list of vectors to check.
    :return: :code:`True` if :code:`vec_list` are mutually orthogonal, and
             :code:`False` otherwise.

    """
    if len(vec_list) <= 1:
        raise ValueError("There must be at least two vectors provided as input.")

    # Convert list of vectors to a 2D array (each vector is a column)
    mat = np.column_stack(vec_list)

    # Compute the matrix of inner products
    inner_product_matrix = np.dot(mat.T.conj(), mat)

    # The diagonal elements will be non-zero (norm of each vector)
    # Set the diagonal elements to zero for the comparison
    np.fill_diagonal(inner_product_matrix, 0)

    # Check if all off-diagonal elements are close to zero
    return np.allclose(inner_product_matrix, 0)
