"""Produces a depolarizng channel."""
import numpy as np
from scipy.sparse import identity
from toqito.states.states.max_entangled import max_entangled


def depolarizing(dim: int, param_p: int = 0) -> np.ndarray:
    r"""
    Produce the depolarizng channel [WIKDC]_.

    The depolarizng channel is the Choi matrix of the completely depolarizng
    channel that acts on `dim`-by-`dim` matrices.

    Produces the partially depolarizng channel `(1-P)*D + P*ID` where `D` is
    the completely depolarizing channel and `ID` is the identity channel.

    Examples
    ==========

    The depolarizing channel maps every density matrix to the maximally-mixed
    state. For example, consider the density operator

    .. math::
        \rho = \frac{1}{2} \begin{pmatrix}
                             1 & 0 & 0 & 1 \\
                             0 & 0 & 0 & 0 \\
                             0 & 0 & 0 & 0 \\
                             1 & 0 & 0 & 1
                           \end{pmatrix}

    corresponding to one of the Bell states. Applying the depolarizing channel
    to :math:`\rho` we have that

    .. math::
        \Phi(\rho) = \frac{1}{4} \begin{pmatrix}
                                    \frac{1}{2} & 0 & 0 & \frac{1}{2} \\
                                    0 & 0 & 0 & 0 \\
                                    0 & 0 & 0 & 0 \\
                                    \frac{1}{2} & 0 & 0 & \frac{1}{2}
                                 \end{pmatrix}.

    This can be observed in `toqito` as follows.

    >>> from toqito.maps.apply_map import apply_map
    >>> from toqito.channels.depolarizing import depolarizing
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    >>> )
    >>> apply_map(test_input_mat, depolarizing(4))
    matrix([[0.125, 0.   , 0.   , 0.125],
            [0.   , 0.   , 0.   , 0.   ],
            [0.   , 0.   , 0.   , 0.   ],
            [0.125, 0.   , 0.   , 0.125]])

    References
    ==========
    .. [WIKDC] Wikipedia: Quantum depolarizing channel
        https://en.wikipedia.org/wiki/Quantum_depolarizing

    :param dim: The dimensionality on which the channel acts.
    :param param_p: Default 0.
    :return:
    """
    # Compute the Choi matrix of the depolarizng channel.

    # Gives a sparse non-normalized state.
    psi = max_entangled(dim=dim, is_sparse=False, is_normalized=False)
    return (1 - param_p) * identity(dim ** 2) / dim + param_p * (psi * psi.conj().T)
