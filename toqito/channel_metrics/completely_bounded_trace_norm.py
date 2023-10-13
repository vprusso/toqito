"""Compute the completely bounded trace norm of a quantum channel."""
import cvxpy as cp
import numpy as np

from toqito.channel_ops import apply_channel, dual_channel
from toqito.channel_props import is_completely_positive, is_quantum_channel
from toqito.matrix_props import trace_norm


def completely_bounded_trace_norm(phi: np.ndarray) -> float:
    r"""
    Compute the completely bounded trace norm / diamond norm of a quantum channel [WatCBNorm18].
    The algorithm in p.11 of [WatSDP12] with implementation in QETLAB [JohQET] is used.

    References
    ==========
    .. [WatCNorm18] : Watrous, John.
    “The theory of quantum information.” Section 3.3.2: “The completely bounded trace norm”.
    Cambridge University Press, 2018.

    .. [WatSDP09]:   Watrous, John.
    "Simpler semidefinite programs for completely bounded norms"
    https://arxiv.org/pdf/1207.5726.pdf

    .. [JohQET]: Nathaniel Johnston. QETLAB:
    A MATLAB toolbox for quantum entanglement, version 0.9.
    https://github.com/nathanieljohnston/QETLAB/blob/master/DiamondNorm.m
    http://www.qetlab.com, January 12, 2016. doi:10.5281/zenodo.44637

    :raises ValueError: If matrix is not square.
    :param phi: superoperator as choi matrix
    :return: The completely bounded trace norm of the channel
    """
    dim_lx, dim_ly = phi.shape

    if dim_lx != dim_ly:
        raise ValueError(
            "The input and output spaces of the superoperator phi must both be square."
        )

    if is_quantum_channel(phi):
        return 1

    elif is_completely_positive(phi):
        v = apply_channel(np.eye(dim_ly), dual_channel(phi))
        return trace_norm(v)

    dim = int(np.sqrt(dim_lx))
    # SDP
    y0 = cp.Variable([dim_lx, dim_lx], complex=True)
    constraints = [y0 == y0.H]
    constraints += [y0 >> 0]

    y1 = cp.Variable([dim_lx, dim_lx], complex=True)
    constraints += [y1 == y1.H]
    constraints += [y1 >> 0]

    a_var = cp.bmat([[y0, -phi], [-phi.conj().T, y1]])
    constraints += [a_var >> 0]
    objective = cp.Minimize(
        cp.norm(cp.partial_trace(y0, dims=(dim, dim), axis=1))
        + cp.norm(cp.partial_trace(y1, dims=(dim, dim), axis=1))
    )

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return problem.value / 2
