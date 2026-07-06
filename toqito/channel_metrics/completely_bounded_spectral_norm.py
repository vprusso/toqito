"""Computes the completely bounded spectral norm of a quantum channel."""

import numpy as np

from toqito.channel_metrics import completely_bounded_trace_norm
from toqito.channel_ops import dual_channel
from toqito.channel_props.channel_dim import channel_dim


def completely_bounded_spectral_norm(
    phi: np.ndarray, dim: int | list[int] | np.ndarray | None = None
) -> float | np.floating:
    r"""Compute the completely bounded spectral norm of a quantum channel.

    As defined in [@watrous2009semidefinite] and [@qetlablink].

    Args:
        phi: superoperator
        dim: A scalar or vector containing the input and output dimensions of `phi`. This only
            needs to be provided if the input and output dimensions of the channel differ (e.g. a
            channel mapping a qubit to a qutrit), since they cannot be inferred from the Choi
            matrix alone in that case. If the channel maps \(M_m\) to \(M_n\), then `dim` is
            the vector `[m, n]`.

    Returns:
        The completely bounded spectral norm of the channel

    Examples:
        To compute the completely bounded spectral norm of a depolarizing channel:

        ```python exec="1" source="above" result="text"
        from toqito.channels import depolarizing
        from toqito.channel_metrics import completely_bounded_spectral_norm
        # Define the depolarizing channel
        choi_depolarizing = depolarizing(dim=2, param_p=0.2)
        print(completely_bounded_spectral_norm(choi_depolarizing))
        ```

    """
    # The dual channel swaps the input and output spaces, so the dimensions passed along to the
    # completely bounded trace norm are reversed.
    dim_in, dim_out, _ = channel_dim(phi, dim=dim, allow_rect=False, compute_env_dim=False)
    dim_in, dim_out = int(dim_in), int(dim_out)
    return completely_bounded_trace_norm(dual_channel(phi, dims=[dim_in, dim_out]), dim=[dim_out, dim_in])
