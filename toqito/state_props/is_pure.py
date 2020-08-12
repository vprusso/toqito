"""Check if state is pure."""
from typing import List, Union

import numpy as np


def is_pure(state: Union[List[np.ndarray], np.ndarray]) -> bool:
    r"""
    Determine if a given state is pure or list of states are pure [WikIsPure]_.

    A state is said to be pure if it is a density matrix with rank equal to 1. Equivalently, the
    state :math:`\rho` is pure if there exists a unit vector :math:`u` such that:

    .. math::
        \rho = u u^*.

    Examples
    ==========

    Consider the following Bell state:

    .. math::
        u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \in \mathcal{X}.

    The corresponding density matrix of :math:`u` may be calculated by:

    .. math::
        \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \in \text{D}(\mathcal{X}).

    Calculating the rank of :math:`\rho` yields that the :math:`\rho` is a pure state. This can be
    confirmed in :code:`toqito` as follows:

    >>> from toqito.states import bell
    >>> from toqito.state_props import is_pure
    >>> u = bell(0)
    >>> rho = u * u.conj().T
    >>> is_pure(rho)
    True

    It is also possible to determine whether a set of density matrices are pure. For instance, we
    can see that the density matrices corresponding to the four Bell states yield a result of
    :code:`True` indicating that all states provided to the function are pure.

    >>> from toqito.states import bell
    >>> from toqito.state_props import is_pure
    >>> u0, u1, u2, u3 = bell(0), bell(1), bell(2), bell(3)
    >>> rho0 = u0 * u0.conj().T
    >>> rho1 = u1 * u1.conj().T
    >>> rho2 = u2 * u2.conj().T
    >>> rho3 = u3 * u3.conj().T
    >>>
    >>> is_pure([rho0, rho1, rho2, rho3])
    True

    References
    ==========
    .. [WikIsPure] Wikipedia: Quantum state - Pure states
        https://en.wikipedia.org/wiki/Quantum_state#Pure_states

    :param state: The density matrix representing the quantum state or a list
                  of density matrices representing quantum states.
    :return: :code:`True` if state is pure and :code:`False` otherwise.
    """
    # Allow the user to enter a list of states to check.
    if isinstance(state, list):
        for rho in state:
            eigs, _ = np.linalg.eig(rho)
            if not np.allclose(np.max(np.diag(eigs)), 1):
                return False
        return True

    # Otherwise, if the user just put in a single state, check that.
    eigs, _ = np.linalg.eig(state)
    return np.allclose(np.max(np.diag(eigs)), 1)
