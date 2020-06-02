"""Purity metric."""
import numpy as np

from toqito.matrix_props import is_density


def purity(rho: np.ndarray) -> float:
    r"""
    Compute the purity of a quantum state [WikPurity]_.

    Examples
    ==========

    Consider the following scaled state defined as the scaled identity matrix

    .. math::
        \rho = \frac{1}{4} \begin{pmatrix}
                         1 & 0 & 0 & 0 \\
                         0 & 1 & 0 & 0 \\
                         0 & 0 & 1 & 0 \\
                         0 & 0 & 0 & 1
                       \end{pmatrix} \in \text{D}(\mathcal{X}).

    Calculating the purity of :math:`\rho` yields :math:`\frac{1}{4}`. This can be observed using
    :code:`toqito` as follows.

    >>> from toqito.state_metrics import purity
    >>> import numpy as np
    >>> purity(np.identity(4) / 4)
    0.25

    References
    ==========
    .. [WikPurity] Wikipedia: Purity (quantum mechanics)
        https://en.wikipedia.org/wiki/Purity_(quantum_mechanics)

    :param rho: Density operator.
    :return: The purity of the quantum state :code:`rho` (i.e., `gamma` is the)
             quantity `np.trace(rho**2)`.
    """
    if not is_density(rho):
        raise ValueError("Purity is only defined for density operators.")
    # "np.real" get rid of the close-to-0 imaginary part.
    return np.real(np.trace(rho ** 2))
