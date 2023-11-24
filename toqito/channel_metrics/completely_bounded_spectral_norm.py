"""Compute the completely bounded spectral norm of a quantum channel."""
import numpy as np

from toqito.channel_metrics import completely_bounded_trace_norm
from toqito.channel_ops import dual_channel


def completely_bounded_spectral_norm(phi: np.ndarray) -> float:
    r"""Compute the completely bounded spectral norm of a quantum channel.
    
    :cite:`Watrous_2009_semidefinite, QETLAB_link`.

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param phi: superoperator
    :return: The completely bounded spectral norm of the channel
    """
    return completely_bounded_trace_norm(dual_channel(phi))
