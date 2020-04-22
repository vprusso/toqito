"""Determines whether or not a channel is completely positive."""
from typing import List, Union
import numpy as np

from toqito.channels.operations.kraus_to_choi import kraus_to_choi
from toqito.channels.properties.is_herm_preserving import is_herm_preserving
from toqito.linear_algebra.properties.is_psd import is_psd


def is_completely_positive(
    phi: Union[np.ndarray, List[List[np.ndarray]]], tol: float = 1e-05
) -> bool:
    r"""
    Determine whether the given channel is completely positive [WatCP18]_.

    A map :math:`\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)` is
    completely positive if it holds that

    .. math::
        \Phi \otimes \mathbb{I}_{\text{L}(\mathcal{Z})

    is a positive map for every complex Euclidean space :math:`\mathcal{Z}`.

    Alternatively, a channel is positive if the corresponding Choi matrix of the
    channel is both Hermitian-preserving and positive semidefinite.

    Examples
    ==========

    We can specify the input as a list of Kraus operators. Consider the map
    :math:`\Phi` defined as

    .. math::
        \Phi(X) = X - U X U^*

    where

    .. math::
        U = \frac{1}{\sqrt{2}}
        \begin{pmatrix}
            1 & 1 \\
            -1 & 1
        \end{pmatrix}.

    This map is not completely positive, as we can verify as follows.

    >>> from toqito.channels.properties.is_completely_positive import is_completely_positive
    >>> import numpy as np
    >>> unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    >>> kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]
    >>> is_completely_positive(kraus_ops)
    False

    We can also specify the input as a Choi matrix. For instance, consider the
    Choi matrix corresponding to the :math:`2`-dimensional completely
    depolarizing channel

    .. math::
        \frac{1}{2}
        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 1
        \end{pmatrix}.

    We may verify that this channel is completely positive

    >>> from toqito.channels.channels.depolarizing import depolarizing
    >>> from toqito.channels.properties.is_completely_positive import is_completely_positive
    >>> is_completely_positive(depolarizing(2))
    True

    References
    ==========
    .. [WatH18] Watrous, John.
        "The theory of quantum information."
        Section: "Linear maps of square operators".
        Cambridge University Press, 2018.

    :param phi: The channel provided as either a Choi matrix or a list of
                Kraus operators.
    :param tol: The tolerance parameter to determine Hermiticity.
    :return: True if the channel is Hermitian-preserving, and False otherwise.
    """
    # If the variable `phi` is provided as a list, we assume this is a list
    # of Kraus operators.
    if isinstance(phi, list):
        phi = kraus_to_choi(phi)

    # Use Choi's theorem to determine whether `phi` is completely positive.
    return is_herm_preserving(phi, tol) and is_psd(phi, tol)
