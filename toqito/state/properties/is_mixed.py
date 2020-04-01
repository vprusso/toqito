"""Determines if state is mixed."""
import numpy as np
from toqito.state.properties.is_pure import is_pure


def is_mixed(state: np.ndarray) -> bool:
    """
    Determine if a given quantum state is mixed.

    A mixed state by definition is a state that is not pure.

    References:
        [1] Wikipedia: Quantum state - Mixed states
        https://en.wikipedia.org/wiki/Quantum_state#Mixed_states

    :param state: The density matrix representing the quantum state.
    :return: True if state is mixed and False otherwise.
    """
    return not is_pure(state)
