"""Computes the completely bounded spectral norm of a quantum channel."""

import numpy as np

from toqito.channel_metrics import completely_bounded_trace_norm
from toqito.channel_ops import dual_channel


def completely_bounded_spectral_norm(phi: np.ndarray) -> float | np.floating:
    r"""Compute the completely bounded spectral norm of a quantum channel.

    As defined in [@Watrous_2009_Semidefinite] and [@QETLAB_link].

    Examples:
        To compute the completely bounded spectral norm of a depolarizing channel:

        ```python exec="1" source="above"
        from toqito.channels import depolarizing
        from toqito.channel_metrics import completely_bounded_spectral_norm
        # Define the depolarizing channel
        choi_depolarizing = depolarizing(dim=2, param_p=0.2)
        print(completely_bounded_spectral_norm(choi_depolarizing))
        ```

    Args:
        phi: superoperator
    
    Returns:
        The completely bounded spectral norm of the channel
    """
    return completely_bounded_trace_norm(dual_channel(phi))
