"""Computes the purity of a quantum state."""
import numpy as np


def purity(rho: np.ndarray) -> float:
    """
    Computes the purity of a quantum state.

    :param rho: A density matrix.
    :return: The purity of the quantum state RHO (i.e., GAMMA is the)
             quantity trace(RHO^2).
    """
    # "np.real" get rid of the close-to-0 imaginary part.
    return np.real(np.trace(rho**2))
