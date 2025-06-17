"""Computes the completely bounded trace norm of a quantum channel."""

import numpy as np
import picos as pc

from toqito.channel_ops import apply_channel, dual_channel
from toqito.channel_props import is_completely_positive, is_quantum_channel
from toqito.matrix_props import trace_norm


def completely_bounded_trace_norm(phi: np.ndarray, solver: str = "cvxopt", **kwargs) -> float:
    r"""Find the completely bounded trace norm of a quantum channel.

    Also known as the diamond norm of a quantum
    channel (Section 3.3.2 of :footcite:`Watrous_2018_TQI`). The algorithm in p.11 of :footcite:`Watrous_2012_Simpler`
    with implementation in QETLAB :footcite:`QETLAB_link` is used.

    Examples
    ========
    To computer the completely bounded spectral norm of a depolarizing channel,

    .. jupyter-execute::

        from toqito.channels import depolarizing
        from toqito.channel_metrics import completely_bounded_trace_norm
        # Define the depolarizing channel
        choi_depolarizing = depolarizing(dim=2, param_p=0.2)
        completely_bounded_trace_norm(choi_depolarizing)


    References
    ==========
    .. footbibliography::



    :raises ValueError: If matrix is not square.
    :param phi: superoperator as choi matrix
    :param solver: Optimization option for `picos` solver. Default option is `solver="cvxopt"`.
    :param kwargs: Additional arguments to pass to picos' solve method.
    :return: The completely bounded trace norm of the channel

    """
    dim_lx, dim_ly = phi.shape

    if dim_lx != dim_ly:
        raise ValueError("The input and output spaces of the superoperator phi must both be square.")

    if is_quantum_channel(phi):
        return 1

    if is_completely_positive(phi):
        v = apply_channel(np.eye(dim_ly), dual_channel(phi))
        return trace_norm(v)

    dim = round(np.sqrt(dim_lx))
    # SDP
    sdp = pc.Problem()
    y0 = pc.HermitianVariable("y0", (dim_lx, dim_lx))
    sdp.add_constraint(y0 >> 0)

    y1 = pc.HermitianVariable("y1", (dim_lx, dim_lx))
    sdp.add_constraint(y1 >> 0)

    a_var = pc.block([[y0, -phi], [-phi.conj().T, y1]])
    sdp.add_constraint(a_var >> 0)
    obj = pc.SpectralNorm(y0.partial_trace(1, dimensions=dim)) + pc.SpectralNorm(y1.partial_trace(1, dimensions=dim))
    sdp.set_objective("min", obj)
    sdp.solve(solver=solver, **kwargs)

    return sdp.value / 2
