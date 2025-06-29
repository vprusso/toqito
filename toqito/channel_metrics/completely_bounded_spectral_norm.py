"""Computes the completely bounded spectral norm of a quantum channel."""

import numpy as np

from toqito.channel_metrics import completely_bounded_trace_norm
from toqito.channel_ops import dual_channel


def completely_bounded_spectral_norm(phi: np.ndarray) -> float:
    r"""Compute the completely bounded spectral norm of a quantum channel.

    As defined in :footcite:`Watrous_2009_Semidefinite` and :footcite:`QETLAB_link`.

    Examples
    ========
    To computer the completely bounded spectral norm of a depolarizing channel,

    .. jupyter-execute::

        from toqito.channels import depolarizing
        from toqito.channel_metrics import completely_bounded_spectral_norm
        # Define the depolarizing channel
        choi_depolarizing = depolarizing(dim=2, param_p=0.2)
        completely_bounded_spectral_norm(choi_depolarizing)

    References
    ==========
    .. footbibliography::


    :param phi: superoperator
    :return: The completely bounded spectral norm of the channel

    """
    return completely_bounded_trace_norm(dual_channel(phi))
