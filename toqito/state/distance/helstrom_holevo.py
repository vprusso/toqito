"""Computes the Helstrom-Holevo distance between two states."""
import numpy as np
from toqito.state.distance.trace_norm import trace_norm


def helstrom_holevo(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Compute the Helstrom-Holevo distance between states `rho` and `sigma`.

    In general, the best success probability to discriminate
    two mixed states represented by :math:`\rho_1` and :math:`\rho_2` is given
    by 1.

    In general, the best success probability to discriminate two mixed states
    represented by :math:`\rho` and :math:`\sigma` is given by

    .. math::

         \frac{1}{2}+\frac{1}{2} \left(\frac{1}{2} \left|\rho - \sigma
         \right|_1\right).

    References:
        [1] Wikipedia: Holevo's theorem.
            https://en.wikipedia.org/wiki/Holevo%27s_theorem

    :param rho: An input matrix.
    :param sigma: An input matrix.
    :return: The Helstrom-Holevo distance between `rho` and `sigma`.
    """
    return 1 / 2 + 1 / 2 * (trace_norm(rho - sigma)) / 2
