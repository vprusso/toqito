"""Fourier matrix."""
import numpy as np


def fourier(dim: int) -> np.ndarray:
    r"""
    Generate the Fourier transform matrix [WikDFT]_.

    Generates the `dim`-by-`dim` unitary matrix that implements the quantum
    Fourier transform.

    The Fourier matrix is defined as:

    .. math::
        W_N = \frac{1}{N}
        \begin{pmatrix}
            1 & 1 & 1 & 1 & \ldots & 1 \\
            1 & \omega & \omega^2 & \omega^3 & \ldots & \omega^{N-1} \\
            1 & \omega^2 & \omega^4 & \omega^6 & \ldots & \omega^{2(N-1)} \\
            1 & \omega^3 & \omega^6 & \omega^9 & \ldots & \omega^{3(N-1)} \\
            \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
            1 & \omega^{N-1} & \omega^{2(N-1)} & \omega^{3(N-1)} &
            \ldots & \omega^{3(N-1)}
        \end{pmatrix}

    Examples
    ==========

    The Fourier matrix generated from :math:`d = 3` yields the following
    matrix:

    .. math::
        W_3 = \frac{1}{3}
        \begin{pmatrix}
            1 & 1 & 1 \\
            0 & \omega & \omega^2 \\
            1 & \omega^2 & \omega^4
        \end{pmatrix}

    >>> from toqito.matrices import fourier
    >>> fourier(3)
    [[ 0.57735027+0.j ,  0.57735027+0.j ,  0.57735027+0.j ],
     [ 0.57735027+0.j , -0.28867513+0.5j, -0.28867513-0.5j],
     [ 0.57735027+0.j , -0.28867513-0.5j, -0.28867513+0.5j]]

    References
    ==========
    .. [WikDFT] Wikipedia: DFT matrix,
        https://en.wikipedia.org/wiki/DFT_matrix

    :param dim: The size of the Fourier matrix.
    :return: The Fourier matrix of dimension `dim`.
    """
    # Primitive root of unity.
    root_unity = np.exp(2 * 1j * np.pi / dim)
    entry_1 = np.arange(0, dim)[:, None]
    entry_2 = np.arange(0, dim)
    return np.power(root_unity, entry_1 * entry_2) / np.sqrt(dim)
