"""Generates a Fourier matrix."""

import numpy as np


def fourier(dim: int) -> np.ndarray:
    r"""Generate the Fourier transform matrix [@WikiDFT].

    Generates the `dim`-by-`dim` unitary matrix that implements the
    quantum Fourier transform.

    The Fourier matrix is defined as:

    \[
        W_N = \frac{1}{\sqrt{N}}
        \begin{pmatrix}
            1 & 1 & 1 & 1 & \ldots & 1 \\
            1 & \omega & \omega^2 & \omega^3 & \ldots & \omega^{N-1} \\
            1 & \omega^2 & \omega^4 & \omega^6 & \ldots & \omega^{2(N-1)} \\
            1 & \omega^3 & \omega^6 & \omega^9 & \ldots & \omega^{3(N-1)} \\
            \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
            1 & \omega^{N-1} & \omega^{2(N-1)} & \omega^{3(N-1)} &
            \ldots & \omega^{(N-1)(N-1)}
        \end{pmatrix}
    \]

    Examples:
        The Fourier matrix generated from \(d = 3\) yields the following matrix:

        \[
            W_3 = \frac{1}{\sqrt{3}}
            \begin{pmatrix}
                1 & 1 & 1 \\
                1 & \omega & \omega^2 \\
                1 & \omega^2 & \omega^4
            \end{pmatrix}
        \]

        ```python exec="1" source="above"
        from toqito.matrices import fourier

        print(fourier(3))
        ```

    Args:
        dim: The size of the Fourier matrix.

    Returns:
        The Fourier matrix of dimension `dim`.

    """
    # Primitive root of unity.
    root_unity = np.exp(2 * 1j * np.pi / dim)
    entry_1 = np.arange(0, dim)[:, None]
    entry_2 = np.arange(0, dim)
    return np.power(root_unity, entry_1 * entry_2) / np.sqrt(dim)
