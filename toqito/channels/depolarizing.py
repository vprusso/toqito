"""The depolarizing channel."""
import numpy as np

from toqito.states import max_entangled


def depolarizing(dim: int, param_p: float = 0) -> np.ndarray:
    r"""
    Produce the partially depolarizing channel [WikDepo]_, [WatDepo18]_.

    The Choi matrix of the completely depolarizing channel that acts on
    :code:`dim`-by-:code:`dim` matrices.

    The *completely depolarizing channel* is defined as

    .. math::
        \Omega(X) = \text{Tr}(X) \omega

    for all :math:`X \in \text{L}(\mathcal{X})`, where

    .. math::
        \omega = \frac{\mathbb{I}_{\mathcal{X}}}{\text{dim}(\mathcal{X})}

    denotes the completely mixed stated defined with respect to the space :math:`\mathcal{X}`.

    Examples
    ==========

    The completely depolarizing channel maps every density matrix to the maximally-mixed state.
    For example, consider the density operator

    .. math::
        \rho = \frac{1}{2} \begin{pmatrix}
                             1 & 0 & 0 & 1 \\
                             0 & 0 & 0 & 0 \\
                             0 & 0 & 0 & 0 \\
                             1 & 0 & 0 & 1
                           \end{pmatrix}

    corresponding to one of the Bell states. Applying the depolarizing channel to :math:`\rho` we
    have that

    .. math::
        \Phi(\rho) = \frac{1}{4} \begin{pmatrix}
                                    \frac{1}{2} & 0 & 0 & \frac{1}{2} \\
                                    0 & 0 & 0 & 0 \\
                                    0 & 0 & 0 & 0 \\
                                    \frac{1}{2} & 0 & 0 & \frac{1}{2}
                                 \end{pmatrix}.

    This can be observed in :code:`toqito` as follows.

    >>> from toqito.channel_ops import apply_channel
    >>> from toqito.channels import depolarizing
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    >>> )
    >>> apply_channel(test_input_mat, depolarizing(4))
    [[0.125 0.    0.    0.125]
     [0.    0.    0.    0.   ]
     [0.    0.    0.    0.   ]
     [0.125 0.    0.    0.125]]

    >>> from toqito.channel_ops import apply_channel
    >>> from toqito.channels import depolarizing
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> )
    >>> apply_channel(test_input_mat, depolarizing(4, 0.5))
    [[17.125  0.25   0.375  0.5  ]
     [ 0.625 17.75   0.875  1.   ]
     [ 1.125  1.25  18.375  1.5  ]
     [ 1.625  1.75   1.875 19.   ]]


    References
    ==========
    .. [WikDepo] Wikipedia: Quantum depolarizing channel
        https://en.wikipedia.org/wiki/Quantum_depolarizing

    .. [WatDepo18] Watrous, John.
        "The theory of quantum information."
        Section: "Replacement channels and the completely depolarizing channel".
        Cambridge University Press, 2018.

    :param dim: The dimensionality on which the channel acts.
    :param param_p: Default 0.
    :return: The Choi matrix of the completely depolarizing channel.
    """
    # Compute the Choi matrix of the depolarizing channel.

    # Gives a sparse non-normalized state.
    psi = max_entangled(dim=dim, is_sparse=False, is_normalized=False)
    return (1 - param_p) * np.identity(dim ** 2) / dim + param_p * (psi * psi.conj().T)
