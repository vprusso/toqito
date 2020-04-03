"""Computes the trace distance of two matrices."""
import numpy as np
from toqito.state.distance.trace_norm import trace_norm


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute the trace distance between density operators `rho` and `sigma`.

    References:
            [1] Quantiki: Trace distance
            https://www.quantiki.org/wiki/trace-distance

    :param rho: An input matrix.
    :param sigma: An input matrix.
    :return: The trace distance between `rho` and `sigma`.
    """
    return trace_norm(rho - sigma) / 2
