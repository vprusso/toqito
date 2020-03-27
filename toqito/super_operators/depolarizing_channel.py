"""Produces a depolarizng channel."""
import numpy as np
from scipy.sparse import identity
from toqito.state.states.max_entangled import max_entangled


def depolarizing_channel(dim: int, param_p: int = 0) -> np.ndarray:
    """
    Produce the depolarizng channel.

    The depolarizng channel is the Choi matrix of the completely depolarizng
    channel that acts on `dim`-by-`dim` matrices.

    Produces the partially depolarizng channel `(1-P)*D + P*ID` where `D` is
    the completely depolarizing channel and `ID` is the identity channel.

    References:
        [1] Wikipedia: Quantum depolarizing channel
        https://en.wikipedia.org/wiki/Quantum_depolarizing_channel

    :param dim: The dimensionality on which the channel acts.
    :param param_p: Default 0.
    """
    # Compute the Choi matrix of the depolarizng channel.

    # Gives a sparse non-normalized state.
    psi = max_entangled(dim=dim, is_sparse=False, is_normalized=False)
    return (1 - param_p) * identity(dim ** 2) / dim + param_p * (psi * psi.conj().T)
