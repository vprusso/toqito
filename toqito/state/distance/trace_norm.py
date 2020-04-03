"""Computes the trace norm of a matrix."""
import numpy as np


def trace_norm(rho: np.ndarray) -> float:
    """
    Compute the trace norm of the matrix `rho`.

    The trace norm :math:`||\rho||_1` of a density matrix :math:`\rho` is the
    sum of the singular values of :math:`\rho`. The singular values are the
    roots of the eignvalues of :math:`\rho \rho^*`.

    References:
        [1] Quantiki: Trace norm
        https://www.quantiki.org/wiki/trace-norm

    :param rho: The input matrix.
    :return: The trace norm of `rho`.
    """
    return np.linalg.norm(rho, ord="nuc")
