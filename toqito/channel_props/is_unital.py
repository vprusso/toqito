"""Determine whether channel is unital."""
import numpy as np

from toqito.channel_ops import apply_channel
from toqito.helper import channel_dim
from toqito.matrix_props import is_identity


def is_unital(
    phi: np.ndarray | list[list[np.ndarray]],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    dim: int | list[int] | np.ndarray = None,
) -> bool:
    r"""Determine whether the given channel is unital.

    A map :math:`\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)` is *unital* if it holds that:

    .. math::
        \Phi(\mathbb{I}_{\mathcal{X}}) = \mathbb{I}_{\mathcal{Y}}.

    If the input channel maps :math:`M_{r,c}` to :math:`M_{x,y}` then :code:`dim` should be the
    list :code:`[[r,x], [c,y]]`. If it maps :math:`M_m` to :math:`M_n`, then :code:`dim` can simply
    be the vector :code:`[m,n]`.

    More information can be found in Chapter: Unital Channels And Majorization from :cite:`Watrous_2018_TQI`).

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

    Additionally, the channel whose Choi matrix is the depolarizing channel is another example of
    a unital channel.

    >>> from toqito.channels import depolarizing
    >>> from toqito.channel_props import is_unital
    >>>
    >>> choi = depolarizing(4)
    >>> is_unital(choi)
    True

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param phi: The channel provided as either a Choi matrix or a list of Kraus operators.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :param dim: A scalar, vector or matrix containing the input and output dimensions of PHI.
    :return: :code:`True` if the channel is unital, and :code:`False` otherwise.

    """
    dim_in, _, _ = channel_dim(phi, dim=dim, allow_rect=False, compute_env_dim=False)

    # Channel is unital if :code:`mat` is the identity matrix.
    mat = apply_channel(np.identity(dim_in), phi)
    return is_identity(mat, rtol=rtol, atol=atol)
