"""Is channel Trace-preserving."""
from typing import List, Union

import numpy as np

from toqito.matrix_props import is_identity
from toqito.channels import partial_trace


def is_trace_preserving(
    phi: Union[np.ndarray, List[List[np.ndarray]]],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    dim: Union[List[int], np.ndarray] = None,
) -> bool:
    r"""
    Determine whether the given channel is Trace-preserving [WatH18]_.

    A map :math:`\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)` is
    *Trace-preserving* if it holds that

    .. math::
        \text{Tr} \left( \Phi(X) \right) = \text{Tr}\left( X \right)

    for every operator :math:`X \in \text{L}(\mathcal{X})`.

    Given the corresponding Choi matrix of the channel, a neccessary and sufficient condition is

    .. math::
        \text{Tr}_{\mathcal{Y}} \left( J(\Phi) \right) = \mathbb{I}_{\mathcal{X}}

    The dimensions of the subsystems are given by the vector :code:`dim`. By default,
    both subsystems have equal dimension.

    Alternatively, given a list of Kraus operators, a neccessary and sufficient condition is

    .. math::
        \sum_{a \in \Sigma} A_a^* B_a = \mathbb{I}_{\mathcal{X}}

    Examples
    ==========

    The map :math:`\Phi` defined as

    .. math::
        \Phi(X) = X - U X U^*

    is not Trace-preserving, where

    .. math::
        U = \frac{1}{\sqrt{2}}
        \begin{pmatrix}
            1 & 1 \\
            -1 & 1
        \end{pmatrix}.

    >>> import numpy as np
    >>> from toqito.channel_props import is_trace_preserving
    >>> unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    >>> kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]
    >>> is_trace_preserving(kraus_ops)
    False

    As another example, the depolarizing channel is Trace-preserving.

    >>> from toqito.channels import depolarizing
    >>> from toqito.channel_props import is_trace_preserving
    >>> choi_mat = depolarizing(2)
    >>> is_trace_preserving(choi_mat)
    True

    References
    ==========
    .. [WatH18] Watrous, John.
        "The theory of quantum information."
        Section: "Linear maps of square operators".
        Cambridge University Press, 2018.

    :param phi: The channel provided as either a Choi matrix or a list of Kraus operators.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :param dim: Dimension of the subsystems. If :code:`None`, all dimensions are assumed to be
                equal.
    :return: True if the channel is Trace-preserving, and False otherwise.
    """
    # If the variable `phi` is provided as a list, we assume this is a list
    # of Kraus operators.
    if isinstance(phi, list):
        phi_l = [A for A, _ in phi]
        phi_r = [B for _, B in phi]

        k_l = np.concatenate(phi_l, axis=0)
        k_r = np.concatenate(phi_r, axis=0)

        mat = k_l.conj().T @ k_r
    else:
        mat = partial_trace(input_mat=phi, sys=2, dim=dim)
    return is_identity(mat, rtol=rtol, atol=atol)
