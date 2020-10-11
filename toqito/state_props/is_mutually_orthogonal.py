"""Check if states are mutually orthogonal."""
from typing import Any, List, Union

import numpy as np


def is_mutually_orthogonal(vec_list: List[Union[np.ndarray, List[Union[float, Any]]]]) -> bool:
    r"""
    Check if list of vectors are mutually orthogonal [WikOrthog]_.

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
    .. [WikOrthog] Wikipedia: Orthogonality
        https://en.wikipedia.org/wiki/Orthogonality

    :param vec_list: The list of vectors to check.
    :return: True if :code:`vec_list` are mutually orthogonal, and False otherwise.
    """
    if len(vec_list) <= 1:
        raise ValueError("There must be at least two vectors provided as input.")

    for i, vec_1 in enumerate(vec_list):
        for j, vec_2 in enumerate(vec_list):
            if i != j:
                if not np.isclose(np.inner(vec_1.conj().T, vec_2.conj().T), 0):
                    return False
    return True
