"""Generates the unitary matrix that implements the quantum Fourier
transform."""
import numpy as np


def fourier_matrix(dim: int) -> np.ndarray:
    """
    Generates the DIM-by-DIM unitary matrix that implements the quantum
    Fourier transform.
    :param dim: The size of the Fourier matrix.
    :return: The Fourier matrix of dimension DIM.
    """
    # Primitive root of unity.
    root_unity = np.exp(2*1j*np.pi/dim)
    entry_1 = np.arange(0, dim)[:, None]
    entry_2 = np.arange(0, dim)
    return np.power(root_unity, entry_1 * entry_2) / np.sqrt(dim)
