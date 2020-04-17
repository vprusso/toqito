"""Computes the trace norm of a matrix."""
import numpy as np


def trace_norm(rho: np.ndarray) -> float:
    r"""
    Compute the trace norm of the matrix `rho` [7]_.

    The trace norm :math:`||\rho||_1` of a density matrix :math:`\rho` is the
    sum of the singular values of :math:`\rho`. The singular values are the
    roots of the eigenvalues of :math:`\rho \rho^*`.

    Examples
    ==========

    Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( e_0 \otimes e_0 + e_1 \otimes e_1 \right)
        \in \mathcal{X}.

    The corresponding density matrix of :math:`u` may be calculated by:

    .. math::
        \rho = u u^* = \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \text{D}(\mathcal{X}).

    It can be observed using `toqito` that :math:`||\rho||_1 = 1` as follows.

    >>> from toqito.states.states.bell import bell
    >>> from toqito.states.distance.trace_norm import trace_norm
    >>> rho = bell(0) * bell(0).conj().T
    >>> trace_norm(rho)
    0.9999999999999999

    References
    ==========
    .. [7] Quantiki: Trace norm
        https://www.quantiki.org/wiki/trace-norm

    :param rho: The input matrix.
    :return: The trace norm of `rho`.
    """
    return np.linalg.norm(rho, ord="nuc")
