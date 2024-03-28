"""Compute the set of pretty bad measurements from an ensemble."""
import numpy as np


def pretty_bad_measurement(vectors: list[np.ndarray], probs: list[float]) -> list[np.ndarray]:
    r"""Return the set of pretty bad measurements from a set of vectors and corresponding probabilities.

    This computes the "pretty bad measurement" is defined in :cite:`Hughston_1993_Complete,` and is an analogous idea to
    the "pretty good measurement" from :cite:`McIrvin_2024_Pretty,`. The "pretty bad measurement" is useful in the
    context of state exclusion where the pretty good measurmennt is often used for minimum-error quantum state
    discrimination.

    See Also
    ========
    pretty_good_measurement

    Examples
    ========
    Consider the depolarizing and identity channels in a 2-dimensional space. The depolarizing channel parameter is
    set to 0.2:

    >>> import numpy as np
    >>> from toqito.channels import depolarizing
    >>> from toqito.channel_metrics import diamond_norm
    >>> choi_depolarizing = depolarizing(dim=2, param_p=0.2)
    >>> choi_identity = np.identity(2**2)
    >>> dn = diamond_norm(choi_depolarizing, choi_identity)
    >>> print("Diamond norm between depolarizing and identity channels: ", '%.2f' % dn)
    Diamond norm between depolarizing and identity channels:  -0.00

    References
    ==========
        .. bibliography::
            :filter: docname in docnames


    :raises ValueError: If matrices are not of equal dimension.
    :raises ValueError: If matrices are not square.
    :param choi_1: A 4**N by 4**N matrix (where N is the number of qubits).
    :param choi_2: A 4**N by 4**N matrix (where N is the number of qubits).

    """
    pass
