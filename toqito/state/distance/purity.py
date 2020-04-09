"""Computes the purity of a quantum state."""
import numpy as np


def purity(rho: np.ndarray) -> float:
    r"""
    Compute the purity of a quantum state [4]_.

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

    Calculating the purity of :math:`\rho` yields :math:`\frac{1}{4}`. This can
    be observed using `toqito` as follows.

    >>> from toqito.state.distance.purity import purity
    >>> import numpy as np
    >>> purity(np.identity(4) / 4)
    0.25

    References
    ==========
    .. [4] Wikipedia: Purity (quantum mechanics)
        https://en.wikipedia.org/wiki/Purity_(quantum_mechanics)

    :param rho: A density matrix.
    :return: The purity of the quantum state `rho` (i.e., `gamma` is the)
             quantity `np.trace(rho**2)`.
    """
    # "np.real" get rid of the close-to-0 imaginary part.
    return np.real(np.trace(rho ** 2))
