"""Compute the completely bounded spectral norm of a quantum channel."""
import numpy as np

from toqito.channel_metrics import completely_bounded_trace_norm
from toqito.channel_ops import dual_channel


def completely_bounded_spectral_norm(phi: np.ndarray) -> float:
    r"""
    Compute the completely bounded spectral norm of a quantum channel.

    References
    ==========
    .. [WatSDP09]:   Watrous, John.
        "Semidefinite Programs for Completely Bounded Norms"
        Theory of Computing, 2009
        http://theoryofcomputing.org/articles/v005a011/v005a011.pdf

    .. [NJ]: Nathaniel Johnston. QETLAB:
        A MATLAB toolbox for quantum entanglement, version 0.9.
        https://github.com/nathanieljohnston/QETLAB/blob/master/CBNorm.m
        http://www.qetlab.com, January 12, 2016. doi:10.5281/zenodo.44637

    :param phi: superoperator
    :return: The completely bounded spectral norm of the channel
    """
    return completely_bounded_trace_norm(dual_channel(phi))
