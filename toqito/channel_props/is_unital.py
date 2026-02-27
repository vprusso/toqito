"""Determines if a channel is unital."""

import numpy as np

from toqito.channel_ops import apply_channel
from toqito.channel_props.channel_dim import channel_dim
from toqito.matrix_props import is_identity


def is_unital(
    phi: np.ndarray | list[list[np.ndarray]],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    dim: int | list[int] | np.ndarray | None = None,
) -> bool:
    r"""Determine whether the given channel is unital.

    A map \(\Phi \in \text{T} \left(\mathcal{X}, \mathcal{Y} \right)\) is *unital* if it holds that:

    \[
        \Phi(\mathbb{I}_{\mathcal{X}}) = \mathbb{I}_{\mathcal{Y}}.
    \]

    If the input channel maps \(M_{r,c}\) to \(M_{x,y}\) then `dim` should be the
    list `[[r,x], [c,y]]`. If it maps \(M_m\) to \(M_n\), then `dim` can simply
    be the vector `[m,n]`.

    More information can be found in Chapter: Unital Channels And Majorization from [@Watrous_2018_TQI]).

    Examples:
        Consider the channel whose Choi matrix is the swap operator. This channel is an example of a
        unital channel.

        ```python exec="1" source="above"
        from toqito.perms import swap_operator
        from toqito.channel_props import is_unital

        choi = swap_operator(3)

        print(is_unital(choi))
        ```

        Additionally, the channel whose Choi matrix is the depolarizing channel is another example of
        a unital channel.

        ```python exec="1" source="above"
        from toqito.channels import depolarizing
        from toqito.channel_props import is_unital

        choi = depolarizing(4)

        print(is_unital(choi))
        ```

    Args:
        phi: The channel provided as either a Choi matrix or a list of Kraus operators.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).
        dim: A scalar, vector or matrix containing the input and output dimensions of PHI.

    Returns:
        `True` if the channel is unital, and `False` otherwise.

    """
    dim_in, _, _ = channel_dim(phi, dim=dim, allow_rect=False, compute_env_dim=False)

    # Channel is unital if `mat` is the identity matrix.
    mat = apply_channel(np.identity(dim_in), phi)
    return is_identity(mat, rtol=rtol, atol=atol)
