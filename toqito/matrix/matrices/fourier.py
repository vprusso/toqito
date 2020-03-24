"""Generate unitary matrix that implements the quantum Fourier transform."""
import numpy as np


def fourier(dim: int) -> np.ndarray:
    r"""
    Generate the Fourier transform matrix.

    Generates the `dim`-by-`dim` unitary matrix that implements the quantum
    Fourier transform.

    The Fourier matrix is defined as:

    ..math::
    `
    W = \frac{1}{N}
    \begin{pmatrix}
        1 & 1 & 1 & 1 & \ldots & 1 \\
        1 & \omega & \omega^2 & \omega^3 & \ldots & \omega^{N-1} \\
        1 & \omega^2 & \omega^4 & \omega^6 & \ldots & \omega^{2(N-1)} \\
        1 & \omega^3 & \omega^6 & \omega^9 & \ldots & \omega^{3(N-1)} \\
        \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
        1 & \omega^{N-1} & \omega^{2(N-1)} & \omega^{3(N-1)} &
        \ldots & \omega^{3(N-1)}
    \end{pmatrix}
    `

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
