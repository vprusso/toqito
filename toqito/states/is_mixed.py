"""Determines if state is mixed."""
import numpy as np
from toqito.states.is_pure import is_pure


def is_mixed(state: np.ndarray) -> bool:
    """
    Determines if a given quantum state is mixed.

    :param state: The density matrix representing the quantum state.
    :return: True if state is mixed and False otherwise.
    """
    return not is_pure(state)
