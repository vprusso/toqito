"""Is channel positive."""
from typing import List, Union

import numpy as np

from toqito.channel_ops import kraus_to_choi
from toqito.matrix_props import is_positive_semidefinite


def is_positive(
    phi: Union[np.ndarray, List[List[np.ndarray]]],
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> bool:
    r"""
    Determine whether the given channel is positive [WatPM18]_.

    A map :math:`\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)` is *positive* if it
    holds that

    .. math::
        \Phi(P) \in \text{Pos}(\mathcal{Y})

    for every positive semidefinite operator :math:`P \in \text{Pos}(\mathcal{X})`.

    Alternatively, a channel is positive if the corresponding Choi matrix of the channel is both
    Hermitian-preserving and positive semidefinite.

    Examples
    ==========

    We can specify the input as a list of Kraus operators. Consider the map :math:`\Phi` defined as

    .. math::
        \Phi(X) = X - U X U^*

    where

    .. math::
        U = \frac{1}{\sqrt{2}}
        \begin{pmatrix}
            1 & 1 \\
            -1 & -1
        \end{pmatrix}.

    This map is not completely positive, as we can verify as follows.

    >>> from toqito.channel_props import is_positive
    >>> import numpy as np
    >>> unitary_mat = np.array([[1, 1], [-1, -1]]) / np.sqrt(2)
    >>> kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]
    >>> is_positive(kraus_ops)
    False

    We can also specify the input as a Choi matrix. For instance, consider the Choi matrix
    corresponding to the :math:`4`-dimensional completely depolarizing channel and may verify
    that this channel is positive.

    >>> from toqito.channels import depolarizing
    >>> from toqito.channel_props import is_positive
    >>> is_positive(depolarizing(4))
    True

    References
    ==========
    .. [WatPM18] Watrous, John.
        "The theory of quantum information."
        Section: "Linear maps of square operators".
        Cambridge University Press, 2018.

    :param phi: The channel provided as either a Choi matrix or a list of Kraus operators.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: True if the channel is positive, and False otherwise.
    """
    # If the variable `phi` is provided as a list, we assume this is a list
    # of Kraus operators.
    if isinstance(phi, list):
        phi = kraus_to_choi(phi)
    return is_positive_semidefinite(phi, rtol, atol)
