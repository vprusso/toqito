"""Shared numerical quadrature helpers."""

import numpy as np


def _gauss_legendre_01(m: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute m-point Gauss-Legendre quadrature nodes and weights on [0, 1].

    The nodes and weights are obtained from the standard Gauss-Legendre rule on
    :math:`[-1, 1]` (via ``numpy.polynomial.legendre.leggauss``) and mapped
    affinely onto :math:`[0, 1]`.

    Args:
        m: Number of quadrature points. Must be at least 1.

    Returns:
        Tuple of nodes and weights on the interval :math:`[0, 1]`.

    Raises:
        ValueError: If ``m`` is less than 1.

    """
    if m < 1:
        raise ValueError("m must be at least 1.")
    x, w = np.polynomial.legendre.leggauss(m)
    return 0.5 * (x + 1), 0.5 * w
