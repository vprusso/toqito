"""Is channel unitary."""


import numpy as np

from toqito.channel_ops import choi_to_kraus
from toqito.matrix_props import is_unitary as is_unitary_matrix


def is_unitary(phi: np.ndarray | list[list[np.ndarray]]) -> bool:
    r"""Given a quantum channel, determine if it is unitary.

    (Section 2.2.1: Definitions and Basic Notions Concerning Channels from
    :cite:`Watrous_2018_TQI`).

    Let :math:`\mathcal{X}` be a complex Euclidean space an let :math:`U \in U(\mathcal{X})` be a
    unitary operator. Then a unitary channel is defined as:

    .. math::
        \Phi(X) = U X U^*.

    Examples
    ==========
    The identity channel is one example of a unitary channel:

    .. math::
        U =
        \begin{pmatrix}
            1 & 0 \\
            0 & 1
        \end{pmatrix}.

    We can verify this as follows:

    >>> from toqito.channel_props import is_unitary
    >>> import numpy as np
    >>> kraus_ops = [[np.identity(2), np.identity(2)]]
    >>> is_unitary(kraus_ops)
    True

    We can also specify the input as a Choi matrix. For instance, consider the Choi matrix
    corresponding to the :math:`2`-dimensional completely depolarizing channel.

    .. math::
        \Omega =
        \frac{1}{2}
        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 1
        \end{pmatrix}.

    We may verify that this channel is not a unitary channel.

    >>> from toqito.channels import depolarizing
    >>> from toqito.channel_props import is_unitary
    >>> is_unitary(depolarizing(2))
    False

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param phi: The channel provided as either a Choi matrix or a list of Kraus operators.
    :return: :code:`True` if the channel is a unitary channel, and :code:`False` otherwise.

    """
    # If the variable `phi` is provided as a ndarray, we assume this is a
    # Choi matrix.
    if isinstance(phi, np.ndarray):
        try:
            phi = choi_to_kraus(phi)
        except ValueError:
            # if we fail to obtain a Kraus representation then input/ouput spaces might be
            # non squares or their dimensions are not equal. Hence the channel is not unitary.
            return False

    # If there is a unique Kraus operator and it's a unitary matrix then the channel is unitary.
    if len(phi) != 1:
        return False

    u_mat = phi[0]
    if isinstance(phi[0], list):
        # we enter here if phi is specified as: [[U, U]] or [[U]]
        u_mat = phi[0][0]
        if len(phi[0]) > 2 or (len(phi[0]) == 2 and not np.allclose(phi[0][0], phi[0][1])):
            return False

    return is_unitary_matrix(u_mat)
