"""Check if state is mixed."""
import numpy as np
from toqito.state_props import is_pure


def is_mixed(state: np.ndarray) -> bool:
    r"""
    Determine if a given quantum state is mixed [WikMix]_.

    A mixed state by definition is a state that is not pure.

    Examples
    ==========

    Consider the following density matrix:

    .. math::
        \rho =  \begin{pmatrix}
                    \frac{3}{4} & 0 \\
                    0 & \frac{1}{4}
                \end{pmatrix} \in \text{D}(\mathcal{X}).

    Calculating the rank of :math:`\rho` yields that the :math:`\rho` is a mixed state. This can be
    confirmed in :code:`toqito` as follows:

    >>> from toqito.states import basis
    >>> from toqito.state_props import is_mixed
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    >>> is_mixed(rho)
    True

    References
    ==========
    .. [WikMix] Wikipedia: Quantum state - Mixed states
        https://en.wikipedia.org/wiki/Quantum_state#Mixed_states

    :param state: The density matrix representing the quantum state.
    :return: True if state is mixed and False otherwise.
    """
    return not is_pure(state)
