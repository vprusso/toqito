"""Properties of quantum channels."""
from typing import List, Union

import numpy as np

from toqito.channel_ops import kraus_to_choi
from toqito.matrix_props import is_psd


__all__ = ["is_completely_positive", "is_herm_preserving", "is_positive"]


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

    Alternatively, a channel is completely positive if the corresponding Choi
    matrix of the channel is both Hermitian-preserving and positive
    semidefinite.

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

    >>> from toqito.channel_props import is_completely_positive
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

    >>> from toqito.channels import depolarizing
    >>> from toqito.channel_props import is_completely_positive
    >>> is_completely_positive(depolarizing(2))
    True

    References
    ==========
    .. [WatCP18] Watrous, John.
        "The theory of quantum information."
        Section: "Linear maps of square operators".
        Cambridge University Press, 2018.

    :param phi: The channel provided as either a Choi matrix or a list of
                Kraus operators.
    :param tol: The tolerance parameter to determine complete positivity.
    :return: True if the channel is completely positive, and False otherwise.
    """
    # If the variable `phi` is provided as a list, we assume this is a list
    # of Kraus operators.
    if isinstance(phi, list):
        phi = kraus_to_choi(phi)

    # Use Choi's theorem to determine whether `phi` is completely positive.
    return is_herm_preserving(phi, tol) and is_psd(phi, tol)


def is_herm_preserving(
    phi: Union[np.ndarray, List[List[np.ndarray]]], tol: float = 1e-05
) -> bool:
    r"""
    Determine whether the given channel is Hermitian-preserving [WatH18]_.

    A map :math:`\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)` is
    Hermitian-preserving if it holds that

    .. math::
        \Phi(H) \in \text{Herm}(\mathcal{Y})

    for every Hermitian operator :math:`H \in \text{Herm}(\mathcal{X}`.

    Examples
    ==========

    The map :math:`\Phi` defined as

    .. math::
        \Phi(X) = X - U X U^*

    is Hermitian-preserving, where

    .. math::
        U = \frac{1}{\sqrt{2}}
        \begin{pmatrix}
            1 & 1 \\
            -1 & 1
        \end{pmatrix}.

    >>> import numpy as np
    >>> from toqito.channel_props import is_herm_preserving
    >>> unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    >>> kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]
    >>> is_herm_preserving(kraus_ops)
    True

    We may also verify whether the corresponding Choi matrix of a given map is
    Hermitian-preserving. The swap operator is the Choi matrix of the transpose
    map, which is Hermitian-preserving as can be seen as follows:

    >>> import numpy as np
    >>> from toqito.perms import swap_operator
    >>> from toqito.channel_props import is_herm_preserving
    >>> unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    >>> choi_mat = swap_operator(3)
    >>> is_herm_preserving(choi_mat)
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

    # Phi is Hermiticity-preserving iff its Choi matrix is Hermitian.
    if phi.shape[0] != phi.shape[1]:
        return False
    return np.max(np.max(np.abs(phi - phi.conj().T))) <= tol


def is_positive(
    phi: Union[np.ndarray, List[List[np.ndarray]]], tol: float = 1e-05
) -> bool:
    r"""
    Determine whether the given channel is positive [WatPM18]_.

    A map :math:`\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)` is
    positive if it holds that

    .. math::
        \Phi(P) \in \text{Pos}(\mathcal{Y})

    for every positive semidefinite operator
    :math:`P \in \text{Pos(\mathcal{X})`.

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
            -1 & -1
        \end{pmatrix}.

    This map is not completely positive, as we can verify as follows.

    >>> from toqito.channel_props import is_positive
    >>> import numpy as np
    >>> unitary_mat = np.array([[1, 1], [-1, -1]]) / np.sqrt(2)
    >>> kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]
    >>> is_positive(kraus_ops)
    False

    We can also specify the input as a Choi matrix. For instance, consider the
    Choi matrix corresponding to the :math:`4`-dimensional completely
    depolarizing channel and may verify that this channel is positive.

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

    :param phi: The channel provided as either a Choi matrix or a list of
                Kraus operators.
    :param tol: The tolerance parameter to determine positivity.
    :return: True if the channel is positive, and False otherwise.
    """
    # If the variable `phi` is provided as a list, we assume this is a list
    # of Kraus operators.
    if isinstance(phi, list):
        phi = kraus_to_choi(phi)
    return is_psd(phi, tol)
