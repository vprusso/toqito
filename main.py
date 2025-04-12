"""Computes the completely bounded trace norm of a quantum channel."""

import numpy as np
import picos as pc

from toqito.channel_ops import apply_channel, dual_channel
from toqito.channel_props import is_completely_positive, is_quantum_channel
from toqito.channels.dephasing import dephasing
from toqito.channels.depolarizing import depolarizing
from toqito.matrix_props import trace_norm


def completely_bounded_trace_norm(phi: np.ndarray, solver: str = "cvxopt", **kwargs) -> float:
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

choi_1 = dephasing(2)
choi_2 = depolarizing(2)
print(completely_bounded_trace_norm(choi_1 - choi_2))
