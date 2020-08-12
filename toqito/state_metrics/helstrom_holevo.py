"""Helstrom-Holevo metric."""
import numpy as np

from toqito.matrix_props import is_density
from toqito.state_metrics import trace_norm


def helstrom_holevo(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Compute the Helstrom-Holevo distance between density matrices [WikHeHo]_.

    In general, the best success probability to discriminate two mixed states represented by
    :math:`\rho` and :math:`\sigma` is given by [WikHeHo]_.

    .. math::
         \frac{1}{2}+\frac{1}{2} \left(\frac{1}{2} \left|\rho - \sigma \right|_1\right).

    Examples
    ==========
    Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \in \mathcal{X}.

    The corresponding density matrix of :math:`u` may be calculated by:

    .. math::
        \rho = u u^* = \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \in \text{D}(\mathcal{X}).

    Calculating the Helstrom-Holevo distance of states that are identical yield a value of
    :math:`1/2`. This can be verified in :code:`toqito` as follows.

    >>> from toqito.states import basis
    >>> from toqito.state_metrics import helstrom_holevo
    >>> import numpy as np
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
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
    .. [WikHeHo] Wikipedia: Holevo's theorem.
        https://en.wikipedia.org/wiki/Holevo%27s_theorem

    :param rho: Density operator.
    :param sigma: Density operator.
    :return: The Helstrom-Holevo distance between :code:`rho` and :code:`sigma`.
    """
    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Helstrom-Holevo is only defined for density operators.")
    return 1 / 2 + 1 / 2 * (trace_norm(rho - sigma)) / 2
