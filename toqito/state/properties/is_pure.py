"""Determines if a state is pure or a list of states are pure."""
from typing import List, Union
from numpy import linalg as lin_alg
import numpy as np


def is_pure(state: Union[List[np.ndarray], np.ndarray]) -> bool:
    r"""
    Determine if a given state is pure or list of states are pure.

    A state is said to be pure if it is a density matrix with rank equal to
    1. Equivalently, the state :math: `\rho` is pure if there exists a unit
    vector :math: `u` such that:

    ..math::
        \rho = u u^*

    References:
        [1] Wikipedia: Quantum state - Pure states
        https://en.wikipedia.org/wiki/Quantum_state#Pure_states

    :param state: The density matrix representing the quantum state or a list
                  of density matrices representing quantum states.
    :return: True if state is pure and False otherwise.
    """
    # Allow the user to enter a list of states to check.
    if isinstance(state, list):
        for rho in state:
            eigs, _ = lin_alg.eig(rho)
            if not np.allclose(np.max(np.diag(eigs)), 1):
                return False
        return True

    # Otherwise, if the user just put in a single state, check that.
    eigs, _ = lin_alg.eig(state)
    return np.allclose(np.max(np.diag(eigs)), 1)
