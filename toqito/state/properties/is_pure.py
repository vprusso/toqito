"""Determines if state is pure."""
from numpy import linalg as lin_alg
import numpy as np


def is_pure(state: np.ndarray) -> bool:
    """
    Determines if a given quantum state is pure.

    :param state: The density matrix representing the quantum state.
    :return: True if state is pure and False otherwise.
    """
    eigs, _ = lin_alg.eig(state)
    return np.allclose(np.max(np.diag(eigs)), 1)
