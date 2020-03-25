"""Computes the Helstrom-Holevo distance between two states."""
import numpy as np
from toqito.state.distance.trace_norm import trace_norm


def helstrom_holevo(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Compute the Helstrom-Holevo distance between states `rho` and `sigma`.

    In general, the best success probability to discriminate
    two mixed states represented by ρ1 and ρ2 is given by 1

    In general, the best success probability to discriminate two mixed states
    represented by ρ and σ is given by

    .. math::
         \frac{1}{2}+\frac{1}{2} \left(\frac{1}{2} \norm{\rho - \sigma}_1\right)

    :param rho: An input matrix.
    :param sigma: An input matrix.
    :return: The Helstrom-Holevo distance between `rho` and `sigma`.

    References:
        [1] Wikipedia: Holevo's theorem.
            https://en.wikipedia.org/wiki/Holevo%27s_theorem
    """
    return 1 / 2 + 1 / 2 * (trace_norm(rho - sigma)) / 2
