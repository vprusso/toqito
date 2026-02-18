"""Generates the depolarizing channel."""

import numpy as np

from toqito.channel_ops import apply_channel as apply_op


def depolarizing(
    dim: int,
    param_p: float = 0,
    input_mat: np.ndarray | None = None,
    apply_channel: bool = False,
) -> np.ndarray:
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

     test_input_mat = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])

     depolarizing(4, input_mat=test_input_mat, apply_channel=True)

    .. jupyter-execute::

     import numpy as np
     from toqito.channels import depolarizing

     test_input_mat = np.arange(1, 17).reshape(4, 4)

     depolarizing(4, 0.5, input_mat=test_input_mat, apply_channel=True)



    References
    ==========
    .. footbibliography::




    :param dim: The dimensionality on which the channel acts.
    :param param_p: Depolarizing probability \(p \) \in [0,1] that mixes the input state
                    with the maximally mixed state. Default 0.
    :param input_mat: Optional input matrix to apply the channel to.
    :param apply_channel: If True and input_mat is provided, apply the channel to input_mat.
        If False (default), return the Choi matrix.
    :return: The Choi matrix of the completely depolarizing channel, or the result of
        applying the channel to input_mat.
    :raises ValueError: If `param_p` is outside the interval [0,1], or if apply_channel is True
        but input_mat is None.

    """
    # Compute the Choi matrix of the depolarizing channel.
    if param_p > 1 or param_p < 0:
        raise ValueError("The depolarizing probability must be between 0 and 1.")

    result = np.zeros((dim**2, dim**2), dtype=np.float64)
    np.fill_diagonal(result, (1 - param_p) / dim)

    if param_p != 0.0:
        idx = np.arange(dim) * (dim + 1)
        result[np.ix_(idx, idx)] += param_p

    if apply_channel:
        if input_mat is None:
            raise ValueError("input_mat is required when apply_channel=True")
        return apply_op(input_mat, result)

    return result
