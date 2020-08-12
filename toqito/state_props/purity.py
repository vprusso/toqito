"""State purity."""
import numpy as np

from toqito.matrix_props import is_density


def purity(rho: np.ndarray) -> float:
    r"""
    Compute the purity of a quantum state [WikPurity]_.

    The negativity of a subsystem can be defined in terms of a density matrix :math:`\rho`: The
    purity of a quantum state :math:`\rho` is defined as

    .. math::
        \text{Tr}(\rho^2),

    where :math:`\text{Tr}` is the trace function.

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

    >>> from toqito.state_props import purity
    >>> import numpy as np
    >>> purity(np.identity(4) / 4)
    0.25

    Calculate the purity of the Werner state.
    >>> from toqito.states import werner
    >>> rho = werner(2, 1 / 4)
    >>> purity(rho)
    0.2653

    References
    ==========
    .. [WikPurity] Wikipedia page for purity (quantum mechanics):
        https://en.wikipedia.org/wiki/Purity_(quantum_mechanics)

    :param rho: A density matrix of a pure state vector.
    :return: A value between 0 and 1 that corresponds to the purity of
            :math:`\rho`.
    """
    if not is_density(rho):
        raise ValueError("Purity is only defined for density operators.")
    # "np.real" get rid of the close-to-0 imaginary part.
    return np.real(np.trace(np.linalg.matrix_power(rho, 2)))
