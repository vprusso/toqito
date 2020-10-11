"""Determine whether channel is unital."""
from typing import List, Union

import numpy as np

from toqito.channel_ops import apply_channel, kraus_to_choi
from toqito.matrix_props import is_identity


def is_unital(
    phi: Union[np.ndarray, List[List[np.ndarray]]],
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> bool:
    r"""
    Determine whether the given channel is unital [WatUnital18]_.

    A map :math:`\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)` is *unital* if it
    holds that

    .. math::
        \Phi(\mathbb{I}_{\mathcal{X}}) = \mathbb{I}_{\mathcal{Y}}.

    Examples
    ==========

    Consider the channel whose Choi matrix is the swap operator. This channel is an example of a
    unital channel.

    >>> from toqito.perms import swap_operator
    >>> from toqito.channel_props import is_unital
    >>>
    >>> choi = swap_operator(3)
    >>> is_unital(choi)
    True

    Alternatively, the channel whose Choi matrix is the depolarizing channel is an example of a
    non-unital channel.

    >>> from toqito.channels import depolarizing
    >>> from toqito.channel_props import is_unital
    >>>
    >>> choi = depolarizing(4)
    >>> is_unital(choi)
    False

    References
    ==========
    .. [WatUnital18] Watrous, John.
        "The theory of quantum information."
        Chapter: Unital channels and majorization
        Cambridge University Press, 2018.

    :param phi: The channel provided as either a Choi matrix or a list of Kraus operators.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: :code:`True` if the channel is unital, and :code:`False` otherwise.
    """
    # If the variable `phi` is provided as a list, we assume this is a list of Kraus operators.
    if isinstance(phi, list):
        phi = kraus_to_choi(phi)

    dim = int(np.sqrt(phi.shape[0]))

    # Channel is unital if :code:`mat` is the identity matrix.
    mat = apply_channel(np.identity(dim), phi)
    return is_identity(mat, rtol=rtol, atol=atol)
