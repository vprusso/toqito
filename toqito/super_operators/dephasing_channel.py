"""Produces a dephasing channel."""
import numpy as np
from toqito.state.states.max_entangled import max_entangled


def dephasing_channel(dim: int, param_p: int = 0) -> np.ndarray:
    """
    Produce the dephasing channel.

    The dephasing channel is the Choi matrix of the completely dephasing
    channel that acts on `dim`-by-`dim` matrices.

    Produces the partially dephasing channel `(1-P)*D + P*ID` where `D` is the
    completely dephasing channel and ID is the identity channel.

    :param dim: The dimensionality on which the channel acts.
    :param param_p: Default is 0.
    """
    # Compute the Choi matrix of the dephasing channel.

    # Gives a sparse non-normalized state.
    psi = max_entangled(dim=dim, is_sparse=False, is_normalized=False)
    return (1 - param_p) * np.diag(np.diag(psi * psi.conj().T)) + param_p * (
        psi * psi.conj().T
    )
