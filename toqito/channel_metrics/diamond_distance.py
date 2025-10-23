"""Computes the diamond norm between two quantum channels."""

import numpy as np


def diamond_distance(choi_1: np.ndarray, choi_2: np.ndarray) -> float:
    r"""Return the diamond norm distance between two quantum channels.

    This function is a wrapper around
    :py:func:`~toqito.channel_metrics.completely_bounded_trace_norm.completely_bounded_trace_norm`, in that it returns
    half of the completely bounded trace norm of the difference of its arguments.

    .. note::
        This calculation becomes very slow for 4 or more qubits.


    Examples
    ========
    Consider the depolarizing and identity channels in a 2-dimensional space. The depolarizing channel parameter is
    set to 0.2:

    .. jupyter-execute::

        import numpy as np
        from toqito.channels import depolarizing
        from toqito.channel_metrics import diamond_distance
        choi_depolarizing = depolarizing(dim=2, param_p=0.2)
        choi_identity = np.identity(2**2)
        diamond_distance(choi_depolarizing, choi_identity)


    Similarly, we can compute the diamond norm between the dephasing channel (with parameter 0.3) and the identity
    channel:

    .. jupyter-execute::

        import numpy as np
        from toqito.channels import dephasing
        from toqito.channel_metrics import diamond_distance
        choi_dephasing = dephasing(dim=2)
        choi_identity = np.identity(2**2)
        diamond_distance(choi_dephasing, choi_identity)


    References
    ==========
        .. footbibliography::



    :raises ValueError: If matrices are not of equal dimension.
    :raises ValueError: If matrices are not square.
    :param choi_1: A 4**N by 4**N matrix (where N is the number of qubits).
    :param choi_2: A 4**N by 4**N matrix (where N is the number of qubits).

    """
    from toqito.channel_metrics import completely_bounded_trace_norm  # noqa

    return completely_bounded_trace_norm(choi_1 - choi_2)
