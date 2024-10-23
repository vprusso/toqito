"""Bures angle, also known as Bures arc, Bures length or quantum angle is a distance metric.

The Bures angle metric is a measure of the statistical distance between quantum states.
"""

import numpy as np

from toqito.state_metrics import fidelity


def bures_angle(rho_1: np.ndarray, rho_2: np.ndarray, decimals: int = 10) -> float:
    r"""Compute the Bures angle of two density matrices :cite:`WikiBures`.

    Calculate the Bures angle between two density matrices :code:`rho_1` and :code:`rho_2` defined by:

    .. math::
        \arccos{\sqrt{F (\rho_1, \rho_2)}}

    where :math:`F(\cdot)` denotes the fidelity between :math:`\rho_1` and :math:`\rho_2`. The return is a value between
    :math:`0` and :math:`\pi / 2`, with :math:`0` corresponding to matrices :code:`rho_1 = rho_2` and :math:`\pi / 2`
    corresponding to the case :code:`rho_1` and :code:`rho_2` with orthogonal support.

    Examples
    ==========

    Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \in \mathcal{X}.

    The corresponding density matrix of :math:`u` may be calculated by:

    .. math::
        \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \in \text{D}(\mathcal{X}).

    In the event where we calculate the Bures angle between states that are identical, we should obtain the value of
    :math:`0`. This can be observed in :code:`toqito` as follows.

    >>> from toqito.state_metrics import bures_angle
    >>> import numpy as np
    >>> rho = 1 / 2 * np.array(
    ...     [[1, 0, 0, 1],
    ...      [0, 0, 0, 0],
    ...      [0, 0, 0, 0],
    ...      [1, 0, 0, 1]]
    ... )
    >>> sigma = rho
    >>> bures_angle(rho, sigma)
    np.float64(0.0)

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :raises ValueError: If matrices are not of equal dimension.
    :param rho_1: Density operator.
    :param rho_2: Density operator.
    :param decimals: Number of decimal places to round to (default 10).
    :return: The Bures angle between :code:`rho_1` and :code:`rho_2`.

    """
    # Perform error checking.
    if not np.all(rho_1.shape == rho_2.shape):
        raise ValueError("InvalidDim: `rho_1` and `rho_2` must be matrices of the same size.")
    # Round fidelity to only 10 decimals to avoid error when :code:`rho_1 = rho_2`.
    return np.real(np.arccos(np.sqrt(np.round(fidelity(rho_1, rho_2), decimals))))
