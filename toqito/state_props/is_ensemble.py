"""Checks if a set of quantum states form an ensemble."""

import numpy as np

from toqito.matrix_props import is_positive_semidefinite


def is_ensemble(states: list[np.ndarray]) -> bool:
    r"""Determine if a set of states constitute an ensemble.

    For more info, see Section: Ensemble Of Quantum States from cite:`Watrous_2018_TQI`.

    An ensemble of quantum states is defined by a function

    .. math::
        \eta : \Gamma \rightarrow \text{Pos}(\mathcal{X})

    that satisfies

    .. math::
        \text{Tr}\left( \sum_{a \in \Gamma} \eta(a) \right) = 1.

    Examples
    ==========

    Consider the following set of matrices

    .. math::
        \eta = \left\{ \rho_0, \rho_1 \right\}

    where

    .. math::
        \rho_0 = \frac{1}{2} \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}, \quad
        \rho_1 = \frac{1}{2} \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}.

    The set :math:`\eta` constitutes a valid ensemble.

    >>> from toqito.state_props import is_ensemble
    >>> import numpy as np
    >>> rho_0 = np.array([[0.5, 0], [0, 0]])
    >>> rho_1 = np.array([[0, 0], [0, 0.5]])
    >>> states = [rho_0, rho_1]
    >>> is_ensemble(states)
    True

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param states: The list of states to check.
    :return: :code:`True` if states form an ensemble and :code:`False` otherwise.

    """
    trace_sum = 0
    for state in states:
        trace_sum += np.trace(state)
        # Constraint: All states in ensemble must be positive semidefinite.
        if not is_positive_semidefinite(state):
            return False
    # Constraint: The sum of the traces of all states within the ensemble must
    # be equal to 1.
    return np.allclose(trace_sum, 1)
