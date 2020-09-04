"""Trace norm metric."""
import numpy as np


def trace_norm(rho: np.ndarray) -> float:
    r"""
    Compute the trace norm of the state [WikTn]_.

    The trace norm :math:`||\rho||_1` of a density matrix :math:`\rho` is the sum of the singular
    values of :math:`\rho`. The singular values are the roots of the eigenvalues of
    :math:`\rho \rho^*`.

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

    It can be observed using :code:`toqito` that :math:`||\rho||_1 = 1` as follows.

    >>> from toqito.states import bell
    >>> from toqito.state_metrics import trace_norm
    >>> rho = bell(0) * bell(0).conj().T
    >>> trace_norm(rho)
    0.9999999999999999

    References
    ==========
    .. [WikTn] Quantiki: Trace norm
        https://www.quantiki.org/wiki/trace-norm

    :param rho: Density operator.
    :return: The trace norm of :code:`rho`.
    """
    return np.linalg.norm(rho, ord="nuc")
