"""Computes the diamond norm between two quantum channels."""

import numpy as np


def diamond_distance(
    choi_1: np.ndarray, choi_2: np.ndarray, dim: int | list[int] | np.ndarray | None = None
) -> float | np.floating:
    r"""Return the diamond norm distance between two quantum channels.

    This function is a wrapper around
    [`completely_bounded_trace_norm`]
    [toqito.channel_metrics.completely_bounded_trace_norm.completely_bounded_trace_norm],
    in that it returns the completely bounded trace norm of the difference of its arguments.

    !!! note
        This calculation becomes very slow for 4 or more qubits.


    Args:
        choi_1: A (d_in * d_out) by (d_in * d_out) Choi matrix of the first channel.
        choi_2: A (d_in * d_out) by (d_in * d_out) Choi matrix of the second channel.
        dim: A scalar or vector containing the input and output dimensions of the channels. This
            only needs to be provided if the input and output dimensions of the channels differ
            (e.g. channels mapping a qubit to a qutrit), since they cannot be inferred from the
            Choi matrices alone in that case. If the channels map \(M_m\) to \(M_n\), then
            `dim` is the vector `[m, n]`.

    Raises:
        ValueError: If matrices are not of equal dimension.
        ValueError: If matrices are not square.

    Examples:
        Consider the depolarizing and identity channels in a 2-dimensional space. The depolarizing channel parameter is
        set to 0.2:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channels import depolarizing
        from toqito.channel_metrics import diamond_distance
        choi_depolarizing = depolarizing(dim=2, param_p=0.2)
        choi_identity = depolarizing(dim=2, param_p=1)  # Identity channel Choi matrix
        print(diamond_distance(choi_depolarizing, choi_identity))
        ```

        Similarly, we can compute the diamond norm between the dephasing channel and the identity
        channel:

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channels import dephasing, depolarizing
        from toqito.channel_metrics import diamond_distance
        choi_dephasing = dephasing(dim=2)
        choi_identity = depolarizing(dim=2, param_p=1)  # Identity channel Choi matrix
        print(diamond_distance(choi_dephasing, choi_identity))
        ```

    """
    from toqito.channel_metrics import completely_bounded_trace_norm  # noqa

    return completely_bounded_trace_norm(choi_1 - choi_2, dim=dim)
