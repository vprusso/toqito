"""The dephasing channel."""
import numpy as np

from toqito.states import max_entangled


def dephasing(dim: int, param_p: float = 0) -> np.ndarray:
    r"""
    Produce the partially dephasing channel [WatDeph18]_.

    The Choi matrix of the completely dephasing channel that acts on :code:`dim`-by-:code:`dim`
    matrices.

    Let :math:`\Sigma` be an alphabet and let :math:`\mathcal{X} = \mathbb{C}^{\Sigma}`. The map
    :math:`\Delta \in \text{T}(\mathcal{X})` defined as

    .. math::
        \Delta(X) = \sum_{a \in \Sigma} X(a, a) E_{a,a}

    for every :math:`X \in \text{L}(\mathcal{X})` is defined as the *completely dephasing channel*.

    Examples
    ==========

    The completely dephasing channel maps kills everything off the diagonal. Consider the
    following matrix

    .. math::
        \rho = \begin{pmatrix}
                   1 & 2 & 3 & 4 \\
                   5 & 6 & 7 & 8 \\
                   9 & 10 & 11 & 12 \\
                   13 & 14 & 15 & 16
               \end{pmatrix}.

    Applying the dephasing channel to :math:`\rho` we have that

    .. math::
        \Phi(\rho) = \begin{pmatrix}
                         1 & 0 & 0 & 0 \\
                         0 & 6 & 0 & 0 \\
                         0 & 0 & 11 & 0 \\
                         0 & 0 & 0 & 16
                     \end{pmatrix}.

    This can be observed in :code:`toqito` as follows.

    >>> from toqito.channel_ops import apply_channel
    >>> from toqito.channels import dephasing
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> )
    >>> apply_channel(test_input_mat, dephasing(4))
    [[ 1.,  0.,  0.,  0.],
     [ 0.,  6.,  0.,  0.],
     [ 0.,  0., 11.,  0.],
     [ 0.,  0.,  0., 16.]])

    We may also consider setting the parameter :code:`p = 0.5`.

    >>> from toqito.channel_ops import apply_channel
    >>> from toqito.channels import dephasing
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> )
    >>> apply_channel(test_input_mat, dephasing(4, 0.5))
    [[17.5  0.   0.   0. ]
     [ 0.  20.   0.   0. ]
     [ 0.   0.  22.5  0. ]
     [ 0.   0.   0.  25. ]]

    References
    ==========
    .. [WatDeph18] Watrous, John.
        "The theory of quantum information."
        Section: "The completely dephasing channel".
        Cambridge University Press, 2018.

    :param dim: The dimensionality on which the channel acts.
    :param param_p: Default is 0.
    :return: The Choi matrix of the dephasing channel.
    """
    # Compute the Choi matrix of the dephasing channel.

    # Gives a sparse non-normalized state.
    psi = max_entangled(dim=dim, is_sparse=False, is_normalized=False)
    return (1 - param_p) * np.diag(np.diag(psi * psi.conj().T)) + param_p * (psi * psi.conj().T)
