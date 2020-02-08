"""Computes the trace norm of a matrix."""
import numpy as np


def trace_norm(rho: np.ndarray) -> float:
    """
    Compute the trace norm of the matrix `rho`.

    :param rho: The input matrix.
    :return: The trace norm of `rho`.

    References:
        [1] Quantiki: Trace norm
            https://www.quantiki.org/wiki/trace-norm
    """
    return np.linalg.norm(rho, ord="nuc")
