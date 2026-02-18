"""Generates the reduction channel."""

import numpy as np
from scipy.sparse import identity

from toqito.channel_ops import apply_channel as apply_op
from toqito.states import max_entangled


def reduction(
    dim: int,
    k: int = 1,
    input_mat: np.ndarray | None = None,
    apply_channel: bool = False,
) -> np.ndarray:
    r"""Produce the reduction map or reduction channel :footcite:`WikiReductionCrit`.

    If :code:`k = 1`, this returns the Choi matrix of the reduction map which is a positive map
    on :code:`dim`-by-:code:`dim` matrices. For a different value of :code:`k`, this yields the
    Choi matrix of the map defined by:

    .. math::
        R(X) = k * \text{Tr}(X) * \mathbb{I} - X,

    where :math:`\mathbb{I}` is the identity matrix. This map is :math:`k`-positive.

    Examples
    ==========

    Using :code:`|toqito⟩`, we can generate the :math:`3`-dimensional (or standard) reduction map
    as follows.


    .. jupyter-execute::

     from toqito.channels import reduction

     reduction(3)

    We can also apply the reduction channel to an input matrix:


    .. jupyter-execute::

     import numpy as np
     from toqito.channels import reduction

     test_input_mat = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
     reduction(3, input_mat=test_input_mat, apply_channel=True)

    References
    ==========
    .. footbibliography::


    :param dim: A positive integer (the dimension of the reduction map).
    :param k: If this positive integer is provided, the script will instead return the Choi
              matrix of the following linear map: Phi(X) := K * Tr(X)I - X.
    :param input_mat: Optional input matrix to apply the channel to.
    :param apply_channel: If True and input_mat is provided, apply the channel to input_mat.
        If False (default), return the Choi matrix.
    :return: The reduction map Choi matrix, or the result of applying the channel to input_mat.
    :raises ValueError: If apply_channel is True but input_mat is None.

    """
    psi = max_entangled(dim, False, False)
    identity_matrix = identity(dim**2)
    choi_mat = k * identity_matrix.toarray() - psi @ psi.conj().T

    if apply_channel:
        if input_mat is None:
            raise ValueError("input_mat is required when apply_channel=True")
        return apply_op(input_mat, choi_mat)

    return choi_mat
