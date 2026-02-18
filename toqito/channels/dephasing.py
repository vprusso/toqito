"""Generates the dephasing channel."""

import numpy as np

from toqito.states import max_entangled


def dephasing(
    dim: int,
    param_p: float = 0,
    input_mat: np.ndarray | None = None,
    apply_channel: bool = False,
    return_kraus: bool = False,
) -> np.ndarray | list[np.ndarray]:
    r"""Produce the partially dephasing channel.

    (Section: The Completely Dephasing Channel from :footcite:`Watrous_2018_TQI`).

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

    This can be observed in :code:`|toqito⟩` as follows.

    .. jupyter-execute::

     import numpy as np
     from toqito.channels import dephasing

     test_input_mat = np.arange(1, 17).reshape(4, 4)

     # Using apply_channel parameter:
     dephasing(4, input_mat=test_input_mat, apply_channel=True)


    We may also consider setting the parameter :code:`p = 0.5`.

    .. jupyter-execute::

     import numpy as np
     from toqito.channels import dephasing

     test_input_mat = np.arange(1, 17).reshape(4, 4)

     # Using apply_channel parameter:
     dephasing(4, 0.5, input_mat=test_input_mat, apply_channel=True)


    References
    ==========
    .. footbibliography::



    :param dim: The dimensionality on which the channel acts.
    :param param_p: Dephasing probability in [0,1]. Default is 0.
    :param input_mat: Optional input matrix to apply the channel to.
    :param apply_channel: If True and input_mat is provided, apply the channel
        to input_mat. If False, return the Choi matrix or Kraus operators.
    :param return_kraus: If True, return Kraus operators instead of Choi matrix.
    :return: Choi matrix, Kraus operators, or applied result depending on parameters.

    """
    # Compute the Choi matrix of the dephasing channel.

    # Gives a sparse non-normalized state.
    psi = max_entangled(dim=dim, is_sparse=False, is_normalized=False)
    choi = (1 - param_p) * np.diag(np.diag(psi @ psi.conj().T)) + param_p * (psi @ psi.conj().T)

    # Apply channel if requested
    if apply_channel:
        if input_mat is None:
            raise ValueError("input_mat is required when apply_channel=True")
        from toqito.channel_ops.apply_channel import apply_channel as apply_op

        return apply_op(input_mat, choi)

    # Return Kraus operators if requested
    if return_kraus:
        from toqito.channel_ops.choi_to_kraus import choi_to_kraus

        return choi_to_kraus(choi)

    return choi
