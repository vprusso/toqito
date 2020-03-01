"""Generate unitary matrix that implements the quantum Fourier transform."""
import numpy as np


def fourier_matrix(dim: int) -> np.ndarray:
    """
    Generate the Fourier transform matrix.

    Generates the `dim`-by-`dim` unitary matrix that implements the quantum
    Fourier transform.

    References:
    [1] Wikipedia: DFT matrix,
        https://en.wikipedia.org/wiki/DFT_matrix

    :param dim: The size of the Fourier matrix.
    :return: The Fourier matrix of dimension `dim`.
    """
    # Primitive root of unity.
    root_unity = np.exp(2*1j*np.pi/dim)
    entry_1 = np.arange(0, dim)[:, None]
    entry_2 = np.arange(0, dim)
    return np.power(root_unity, entry_1 * entry_2) / np.sqrt(dim)
