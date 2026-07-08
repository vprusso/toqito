"""Computes the completely bounded trace norm of a quantum channel."""

from typing import Any

import numpy as np
import picos as pc

from toqito.channel_ops import apply_channel, dual_channel
from toqito.channel_props import is_completely_positive, is_trace_preserving
from toqito.channel_props.channel_dim import channel_dim


def completely_bounded_trace_norm(
    phi: np.ndarray,
    solver: str = "cvxopt",
    dim: int | list[int] | np.ndarray | None = None,
    **kwargs: Any,
) -> float | np.floating:
    r"""Find the completely bounded trace norm of a quantum channel.

    Also known as the diamond norm of a quantum channel (Section 3.3.2 of [@watrous2018theory]).
    The algorithm in p.11 of [@watrous2012simpler] with implementation in QETLAB [@qetlablink] is used.

    Args:
        phi: superoperator as choi matrix
        solver: Optimization option for `picos` solver. Default option is `solver="cvxopt"`.
        dim: A scalar or vector containing the input and output dimensions of `phi`. This only
            needs to be provided if the input and output dimensions of the channel differ (e.g. a
            channel mapping a qubit to a qutrit), since they cannot be inferred from the Choi
            matrix alone in that case. If the channel maps \(M_m\) to \(M_n\), then `dim` is
            the vector `[m, n]`.
        kwargs: Additional arguments to pass to picos' solve method.

    Returns:
        The completely bounded trace norm of the channel

    Raises:
        ValueError: If matrix is not square.
        ValueError: If the dimensions of `phi` are inconsistent with `dim` (or, when `dim` is not
            provided, if the input and output dimensions cannot be inferred from the Choi matrix).

    Examples:
        To compute the completely bounded trace norm of a depolarizing channel:

        ```python exec="1" source="above" result="text"
        from toqito.channels import depolarizing
        from toqito.channel_metrics import completely_bounded_trace_norm
        # Define the depolarizing channel
        choi_depolarizing = depolarizing(dim=2, param_p=0.2)
        print(completely_bounded_trace_norm(choi_depolarizing))
        ```

    """
    dim_lx, dim_ly = phi.shape

    if dim_lx != dim_ly:
        raise ValueError("The input and output spaces of the superoperator phi must both be square.")

    # Determine the input and output dimensions of the channel. This correctly handles channels
    # with unequal input/output dimensions (given `dim`) instead of assuming both are
    # sqrt(dim_lx).
    dim_in, dim_out, _ = channel_dim(phi, dim=dim, allow_rect=False, compute_env_dim=False)
    dim_in, dim_out = int(dim_in), int(dim_out)

    if is_completely_positive(phi):
        # Quantum channels have a completely bounded trace norm of 1.
        if is_trace_preserving(phi, dim=[dim_in, dim_out]):
            return 1
        # For a completely positive map, the completely bounded trace norm equals the operator
        # (spectral) norm of the adjoint applied to the identity, ||Phi^*(I_out)||_inf (Watrous,
        # The Theory of Quantum Information, Section 3.3). `dual_channel` gives Phi^*, and the
        # identity is on the output space of dimension `dim_out`.
        v = apply_channel(np.eye(dim_out), dual_channel(phi, dims=[dim_in, dim_out]))
        return np.linalg.norm(v, 2)

    # SDP
    sdp = pc.Problem()
    y0 = pc.HermitianVariable("y0", (dim_lx, dim_lx))
    sdp.add_constraint(y0 >> 0)

    y1 = pc.HermitianVariable("y1", (dim_lx, dim_lx))
    sdp.add_constraint(y1 >> 0)

    a_var = pc.block([[y0, -phi], [-phi.conj().T, y1]])
    sdp.add_constraint(a_var >> 0)
    # The Choi matrix subsystems are ordered as (input, output), so the output space is traced
    # out of each block.
    obj = pc.SpectralNorm(y0.partial_trace(1, dimensions=(dim_in, dim_out))) + pc.SpectralNorm(
        y1.partial_trace(1, dimensions=(dim_in, dim_out))
    )
    sdp.set_objective("min", obj)
    sdp.solve(solver=solver, **kwargs)

    return sdp.value / 2
