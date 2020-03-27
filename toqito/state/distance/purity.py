"""Computes the purity of a quantum state."""
import numpy as np


def purity(rho: np.ndarray) -> float:
    """
    Compute the purity of a quantum state.

    References:
        [1] Wikipedia: Purity (quantum mechanics)
        https://en.wikipedia.org/wiki/Purity_(quantum_mechanics)

    :param rho: A density matrix.
    :return: The purity of the quantum state `rho` (i.e., `gamma` is the)
             quantity `np.trace(rho**2)`.
    """
    # "np.real" get rid of the close-to-0 imaginary part.
    return np.real(np.trace(rho ** 2))
