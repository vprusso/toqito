"""Generates the depolarizing channel."""

import numpy as np

from toqito.states import max_entangled


def depolarizing(dim: int, param_p: float = 0) -> np.ndarray:
    r"""Produce the partially depolarizing channel.

    (Section: Replacement Channels and the Completely Depolarizing Channel from
    :footcite:`Watrous_2018_TQI`).

    The Choi matrix of the completely depolarizing channel :footcite:`WikiDepo` that acts on
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

    This can be observed in :code:`|toqito⟩` as follows.

    .. jupyter-execute::

     import numpy as np
     from toqito.channels import depolarizing
     from toqito.channel_ops import apply_channel

     test_input_mat = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])

     apply_channel(test_input_mat, depolarizing(4))

    .. jupyter-execute::

     import numpy as np
     from toqito.channels import depolarizing
     from toqito.channel_ops import apply_channel

     test_input_mat = np.arange(1, 17).reshape(4, 4)

     apply_channel(test_input_mat, depolarizing(4, 0.5))



    References
    ==========
    .. footbibliography::




    :param dim: The dimensionality on which the channel acts.
    :param param_p: Default 0.
    :return: The Choi matrix of the completely depolarizing channel.

    """
    # Compute the Choi matrix of the depolarizing channel.

    # Gives a sparse non-normalized state.
    psi = max_entangled(dim=dim, is_sparse=False, is_normalized=False)
    return (1 - param_p) * np.identity(dim**2) / dim + param_p * (psi @ psi.conj().T)
