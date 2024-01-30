"""Hilbert-Schmidt metric."""
import numpy as np

from toqito.matrix_props import is_density


def hilbert_schmidt(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""Compute the Hilbert-Schmidt distance between two states :cite:`WikiHilbSchOp`.

    The Hilbert-Schmidt distance between density operators :math:`\rho` and :math:`\sigma` is defined as

    .. math::
        D_{\text{HS}}(\rho, \sigma) = \text{Tr}((\rho - \sigma)^2) = \left\lVert \rho - \sigma
        \right\rVert_2^2.

    Examples
    ==========

    One may consider taking the Hilbert-Schmidt distance between two Bell states. In :code:`toqito`, one may accomplish
    this as

    >>> from toqito.states import bell
    >>> from toqito.state_metrics import hilbert_schmidt
    >>> rho = bell(0) * bell(0).conj().T
    >>> sigma = bell(3) * bell(3).conj().T
    >>> hilbert_schmidt(rho, sigma)
    1

    References
    ==========
    .. bibliography::
        :filter: docname in docnames



    :raises ValueError: If matrices are not density operators.
    :param rho: An input matrix.
    :param sigma: An input matrix.
    :return: The Hilbert-Schmidt distance between :code:`rho` and :code:`sigma`.

    """
    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Hilbert-Schmidt is only defined for density operators.")
    return np.linalg.norm(rho - sigma, ord=2) ** 2
