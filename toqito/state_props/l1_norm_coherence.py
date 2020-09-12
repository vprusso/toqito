"""Compute the l1-norm of coherence of a quantum state."""
import numpy as np
from toqito.state_ops import pure_to_mixed


def l1_norm_coherence(rho: np.ndarray) -> float:
    r"""
    Compute the l1-norm of coherence of a quantum state [RPWL17]_.

    The :math:`\ell_1`-norm of coherence of a quantum state :math:`\rho` is
    defined as

    .. math::
        C_{\ell_1}(\rho) = \sum_{i \not= j} \left|\rho_{i,j}\right|,

    where :math:`\rho_{i,j}` is the :math:`(i,j)^{th}`-entry of :math:`\rho`
    in the standard basis.

    The :math:`\ell_1`-norm of coherence is the sum of the absolute values of
    the sum of the absolute values of the off-diagonal entries of the density
    matrix :code:`rho` in the standard basis.

    This function was adapted from QETLAB.

    Examples
    ========

    The largest possible value of the :math:`\ell_1`-norm of coherence on
    :math:`d`-dimensional states is :math:`d-1`, and is attained exactly by
    the "maximally coherent states": pure states whose entries all have the
    same absolute value.

    >>> from toqito.state_props import l1_norm_coherence
    >>> import numpy as np
    >>>
    >>> # Maximally coherent state.
    >>> v = np.ones((3,1))/np.sqrt(3)
    >>> l1_norm_coherence(v)
    2

    References
    ==========
    .. [RPWL17] "Logarithmic coherence: Operational interpretation of 1-norm coherence"
        Rana, Swapan, Parashar, Preeti, Winter, Andreas, Lewenstein, Maciej
        Physical Review A 96.5 (2017): 052336.
        https://arxiv.org/pdf/1612.09234.pdf

    :param rho: A matrix or vector.
    :return: The l1-norm coherence of :code:`rho`.
    """
    rho = pure_to_mixed(rho)
    return np.sum(np.sum(np.abs(rho))) - np.trace(rho)
