"""Determines if state is mixed."""
import numpy as np
from toqito.states.properties.is_pure import is_pure


def is_mixed(state: np.ndarray) -> bool:
    r"""
    Determine if a given quantum state is mixed [WIKMIX]_.

    A mixed state by definition is a state that is not pure.

    Examples
    ==========

    Consider the following density matrix

    .. math::
        \rho =  \begin{pmatrix}
                    \frac{3}{4} & 0 \\
                    0 & \frac{1}{4}
                \end{pmatrix} \text{D}(\mathcal{X}).

    Calculating the rank of $\rho$ yields that the $\rho$ is a mixed state. This
    can be confirmed in `toqito` as follows:

    >>> from toqito.core.ket import ket
    >>> from toqito.states.properties.is_mixed import is_mixed
    >>> e_0, e_1 = ket(2, 0), ket(2, 1)
    >>> rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    >>> is_mixed(rho)
    True

    References
    ==========
    .. [WIKMIX] Wikipedia: Quantum state - Mixed states
        https://en.wikipedia.org/wiki/Quantum_state#Mixed_states

    :param state: The density matrix representing the quantum state.
    :return: True if state is mixed and False otherwise.
    """
    return not is_pure(state)
