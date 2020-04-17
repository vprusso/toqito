"""Computes the Helstrom-Holevo distance between two states."""
import numpy as np
from toqito.states.distance.trace_norm import trace_norm


def helstrom_holevo(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Compute the Helstrom-Holevo distance between states `rho` and `sigma` [3]_.

    In general, the best success probability to discriminate
    two mixed states represented by :math:`\rho_1` and :math:`\rho_2` is given
    by 1.

    In general, the best success probability to discriminate two mixed states
    represented by :math:`\rho` and :math:`\sigma` is given by

    .. math::

         \frac{1}{2}+\frac{1}{2} \left(\frac{1}{2} \left|\rho - \sigma
         \right|_1\right).

    Examples
    ==========
    Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( e_0 \otimes e_0 + e_1 \otimes e_1 \right)
        \in \mathcal{X}.

    The corresponding density matrix of :math:`u` may be calculated by:

    .. math::
        \rho = u u^* = \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \text{D}(\mathcal{X}).

    Calculating the Helstrom-Holevo distance of states that are identical yield
    a value of :math:`1/2`. This can be verified in `toqito` as follows.

    >>> from toqito.base.ket import ket
    >>> from toqito.states.distance.helstrom_holevo import helstrom_holevo
    >>> import numpy as np
    >>> e_0, e_1 = ket(2, 0), ket(2, 1)
    >>> e_00 = np.kron(e_0, e_0)
    >>> e_11 = np.kron(e_1, e_1)
    >>>
    >>> u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
    >>> rho = u_vec * u_vec.conj().T
    >>> sigma = rho
    >>>
    >>> helstrom_holevo(rho, sigma)
    0.5

    References
    ==========
    .. [3] Wikipedia: Holevo's theorem.
        https://en.wikipedia.org/wiki/Holevo%27s_theorem

    :param rho: An input matrix.
    :param sigma: An input matrix.
    :return: The Helstrom-Holevo distance between `rho` and `sigma`.
    """
    return 1 / 2 + 1 / 2 * (trace_norm(rho - sigma)) / 2
