"""Computes the Hilbert-Schmidt distance between two states."""
import numpy as np


def hilbert_schmidt(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Compute the Hilbert-Schmidt distance between two states [WIKHS]_.


    The Hilbert-Schmidt distance between density operators :math:`\rho` and
    :math:`\sigma` is defined as

    .. math::
        D_{\text{HS})(\rho, \sigma) = \text{Tr}((\rho - \sigma)^2) =
        \left\lVert \rho - \sigma \right\rVert_2^2.

    Examples
    ==========

    Consider the

    References
    ==========
    .. [WIKHS] Wikipedia: Hilbert-Schmidt operator.
        https://en.wikipedia.org/wiki/Hilbert%E2%80%93Schmidt_operator

    :param rho: An input matrix.
    :param sigma: An input matrix.
    :return: The Hilbert-Schmidt distance between `rho` and `sigma`.
    """
    return np.linalg.norm(rho - sigma, ord=2)**2
