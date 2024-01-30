"""Trace distance metric."""
import numpy as np

from toqito.matrix_props import is_density, trace_norm


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""Compute the trace distance between density operators `rho` and `sigma`.

    The trace distance between :math:`\rho` and :math:`\sigma` is defined as

    .. math::
        \delta(\rho, \sigma) = \frac{1}{2} \left( \text{Tr}(\left| \rho - \sigma
         \right| \right).

    More information on the trace distance can be found in :cite:`Quantiki_TrDist`.

    Examples
    ==========

    Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \in \mathcal{X}.

    The corresponding density matrix of :math:`u` may be calculated by:

    .. math::
        \rho = u u^* = \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \in \text{D}(\mathcal{X}).

    The trace distance between :math:`\rho` and another state :math:`\sigma` is equal to :math:`0` if any only if
    :math:`\rho = \sigma`. We can check this using the :code:`toqito` package.

    >>> from toqito.states import bell
    >>> from toqito.matrix_props import trace_norm
    >>> rho = bell(0) * bell(0).conj().T
    >>> sigma = rho
    >>> trace_distance(rho, sigma)
    0.0

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :raises ValueError: If matrices are not of density operators.
    :param rho: An input matrix.
    :param sigma: An input matrix.
    :return: The trace distance between :code:`rho` and :code:`sigma`.

    """
    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Trace distance only defined for density matrices.")
    return trace_norm(np.abs(rho - sigma)) / 2
