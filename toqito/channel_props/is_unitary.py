"""Is channel unitary."""
from typing import List, Union

import numpy as np

from toqito.channel_ops import choi_to_kraus

def is_unitary(
    phi: Union[np.ndarray, List[List[np.ndarray]]]
) -> bool:
    r"""
    Given a quantum channel, determine if it is unitary [WatUC18]_.

    Let :math:`\mathcal{X}` be a complex Euclidean space an let :math:`U \in U(\mathcal{X})` be a
    unitary operator. Then a unitary channel is defined as:

    .. math::
        \Phi(X) = U X U^*

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
    .. [WatUC18] Watrous, John.
        "The Theory of Quantum Information."
        Section: "2.2.1  Definitions and basic notions concerning channels".
        Cambridge University Press, 2018.

    :param phi: The channel provided as either a Choi matrix or a list of Kraus operators.
    :return: :code:`True` if the channel is a unitary channel, and :code:`False` otherwise.
    """
    # If the variable `phi` is provided as a ndarray, we assume this is a
    # choi operator.
    if isinstance(phi, np.ndarray):
        phi = choi_to_kraus(phi)

    # If the length of the list of krauss operarator is equal to one, the channel is unitary.
    return len(phi)==1
