"""Generates the depolarizing channel."""

import numpy as np

from toqito.states import max_entangled


def depolarizing(dim: int, param_p: float = 0) -> np.ndarray:
    r"""Produce the partially depolarizing channel.

    (Section: Replacement Channels and the Completely Depolarizing Channel from
    :cite:`Watrous_2018_TQI`).

    The Choi matrix of the completely depolarizing channel :cite:`WikiDepo` that acts on
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
                                    1 & 0 & 0 & 0 \\
                                    0 & 1 & 0 & 0 \\
                                    0 & 0 & 1 & 0 \\
                                    0 & 0 & 0 & 1
                                 \end{pmatrix}.

    This can be observed in :code:`toqito` as follows.

    >>> from toqito.channel_ops import apply_channel
    >>> from toqito.channels import depolarizing
    >>> import numpy as np
    >>> test_input_mat = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    >>> apply_channel(test_input_mat, depolarizing(4,param_p=0.1))
    array([[0.25, 0.  , 0.  , 0.  ],
           [0.  , 0.25, 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.25]])

    >>> from toqito.channel_ops import apply_channel
    >>> from toqito.channels import depolarizing
    >>> import numpy as np
    >>> test_input_mat = np.array(
    ...     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    ... )
    >>> apply_channel(test_input_mat, depolarizing(4, param_p=0.1))
    array([[8.5, 0. , 0. , 0. ],
           [0. , 8.5, 0. , 0. ],
           [0. , 0. , 8.5, 0. ],
           [0. , 0. , 0. , 8.5]])



    References
    ==========
    .. bibliography::
        :filter: docname in docnames



    :param dim: The dimensionality on which the channel acts.
    :param param_p: A parameter in the range [0, 1] representing the probability of applying the
                 completely depolarizing channel. A value of 0 means no depolarization
                 (identity channel), while a value of 1 means complete depolarization.
    :return: The Choi matrix of the completely depolarizing channel.


    Note:
        This function follows the standard convention where `param_p` controls the amount of
        depolarization applied. If using the QETLAB convention where `param_p` and `1 - param_p`
        are swapped, adjust your input accordingly.

        The convention used here is consistent with standard quantum information references
        such as Nielsen & Chuang. However, it differs from the convention used by QETLAB
        (https://www.qetlab.com/DepolarizingChannel), where the roles of `param_p` and
        `1 - param_p` are reversed.

    """
    # Compute the Choi matrix of the depolarizing channel.

    # Gives a sparse non-normalized state.
    psi = max_entangled(dim=dim, is_sparse=False, is_normalized=False)
    return param_p * np.identity(dim**2) / dim + (1 - param_p) * (psi @ psi.conj().T)
