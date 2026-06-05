"""Calculates the probability of error for channel exclusion (channel antidistinguishability)."""

from typing import Any

import numpy as np
import picos

from toqito.state_opt.state_exclusion import state_exclusion


def channel_exclusion(
    channels: list[np.ndarray],
    probs: list[float] | None = None,
    strategy: str = "min_error",
    solver: str = "cvxopt",
    primal_dual: str = "dual",
    **kwargs: Any,
) -> tuple[float, list[picos.HermitianVariable]]:
    r"""Compute the minimum error probability for channel exclusion.

    The *channel exclusion* problem (also known as channel antidistinguishability) involves a
    collection of :math:`n` quantum channels

    .. math::
        \Phi = \{ \Phi_1, \ldots, \Phi_n \}

    with prior probabilities :math:`p = \{ p_1, \ldots, p_n \}`. One channel :math:`\Phi_i` is
    selected with probability :math:`p_i` and applied to an input state :math:`\rho` chosen by Bob.
    Bob receives the output and must identify which channel was *not* used. This is the
    channel-level analogue of :func:`~toqito.state_opt.state_exclusion`.

    Via the Choi-Jamiołkowski isomorphism, the joint optimization over the input state :math:`\rho`
    and the measurement :math:`\{ M_i \}` reduces to state exclusion on the (normalized) Choi
    matrices :math:`\widetilde{J}(\Phi_i) = J(\Phi_i) / \mathrm{Tr}(J(\Phi_i))`. The min-error
    channel exclusion SDP is therefore equivalent to:

    .. math::
        \begin{equation}
        \begin{aligned}
        \text{minimize:} \quad & \sum_{i=1}^n p_i \langle M_i,\, \widetilde{J}(\Phi_i) \rangle \\
        \text{subject to:} \quad & \sum_{i=1}^n M_i = \mathbb{I}, \\
        & M_1, \ldots, M_n \geq 0,
        \end{aligned}
        \end{equation}

    with dual

    .. math::
        \begin{equation}
        \begin{aligned}
        \text{maximize:} \quad & \mathrm{Tr}(Y) \\
        \text{subject to:} \quad & Y \preceq p_i\, \widetilde{J}(\Phi_i) \quad \forall\, i, \\
        & Y \in \mathrm{Herm}(\mathcal{X}).
        \end{aligned}
        \end{equation}

    Args:
        channels: A list of channels provided as Choi matrices (square numpy arrays).
            All channels must have the same dimensions.
        probs: Respective list of prior probabilities for each channel. If :code:`None`,
            a uniform distribution is assumed.
        strategy: Discrimination strategy. Either :code:`"min_error"` (default) or
            :code:`"unambiguous"`.
        solver: Solver to use for the picos SDP. Default is :code:`"cvxopt"`.
        primal_dual: Whether to solve the :code:`"primal"` or :code:`"dual"` (default) problem.
        **kwargs: Additional keyword arguments passed to :func:`picos.Problem.solve`.

    Returns:
        A tuple :code:`(val, measurements)` where :code:`val` is the optimal exclusion
        probability and :code:`measurements` is the list of optimal measurement operators.

    Raises:
        ValueError: If fewer than 2 channels are provided.
        ValueError: If the number of channels and probabilities do not match.
        ValueError: If channels have inconsistent dimensions.

    Examples:
        Channel exclusion of two identical depolarizing channels. Since the channels are
        indistinguishable, the optimal exclusion probability equals the maximum prior (here
        :math:`1/2`).

        >>> import numpy as np
        >>> from toqito.channel_metrics.channel_exclusion import channel_exclusion
        >>> from toqito.channels import depolarizing
        >>> choi1 = depolarizing(2, param_p=0.5)
        >>> choi2 = depolarizing(2, param_p=0.5)
        >>> val, _ = channel_exclusion([choi1, choi2])
        >>> float(np.around(val, decimals=2))
        0.5

        Channel exclusion of two distinct depolarizing channels.

        >>> import numpy as np
        >>> from toqito.channel_metrics.channel_exclusion import channel_exclusion
        >>> from toqito.channels import depolarizing
        >>> choi1 = depolarizing(2, param_p=0.0)
        >>> choi2 = depolarizing(2, param_p=1.0)
        >>> val, _ = channel_exclusion([choi1, choi2])
        >>> float(np.around(val, decimals=4))
        0.125

    References:
        .. [1] Stratton, B., Hsieh, C.-Y., and Skrzypczyk, P.
           "Operational Interpretation of the Choi Rank Through k-State Exclusion."
           arXiv:2406.08360 (2024).

    """
    if len(channels) < 2:
        raise ValueError("At least 2 channels are required for channel exclusion.")

    n = len(channels)
    probs = [1 / n] * n if probs is None else probs

    if len(probs) != n:
        raise ValueError(
            f"Number of probabilities ({len(probs)}) must match number of channels ({n})."
        )

    # Convert to numpy arrays
    choi_matrices = [np.array(ch, dtype=complex) for ch in channels]

    # Check consistent dimensions
    shapes = [J.shape for J in choi_matrices]
    if len(set(shapes)) != 1:
        raise ValueError(
            f"All channels must have the same Choi matrix dimensions. Got: {shapes}"
        )

    if choi_matrices[0].ndim != 2 or choi_matrices[0].shape[0] != choi_matrices[0].shape[1]:
        raise ValueError("Each channel must be provided as a square Choi matrix.")

    # Normalize Choi matrices so that Tr(J) = 1 (required for the state exclusion reduction)
    normalized = [J / np.trace(J) for J in choi_matrices]

    return state_exclusion(
        vectors=normalized,
        probs=probs,
        strategy=strategy,
        solver=solver,
        primal_dual=primal_dual,
        **kwargs,
    )
