"""Determine probability of obtaining measurement outcome."""
import numpy as np


def measure(measurement: np.ndarray, state: np.ndarray) -> float:
    r"""
    Determine probability of obtaining a measurement outcome applied to state.

    A *measurement* is defined by a function

    .. math::
        \mu : \Sigma \rightarrow \text{Pos}(\mathcal{X}),

    for some choice of alphabet :math:`\Sigma` and a complex Euclidean space :math:`\mathcal{X}`
    that satisfies

    .. math::
        \sum_{a \in \Sigma} \mu(a) = \mathbb{I}_{\mathcal{X}}.

    Further information can be found here [WikMeas]_.

    Examples
    ==========

    Consider the following state:

    .. math::
        u = \frac{1}{\sqrt{3}} e_0 + \sqrt{\frac{2}{3}} e_1

    where we define :math:`u u^* = \rho \in \text{D}(\mathcal{X})`.

    Define measurement operators

    .. math::
        P_0 = e_0 e_0^* \quad \text{and} \quad P_1 = e_1 e_1^*.

    >>> from toqito.states import basis
    >>> from toqito.measurement_ops import measure
    >>> import numpy as np
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>>
    >>> u = 1/np.sqrt(3) * e_0 + np.sqrt(2/3) * e_1
    >>> rho = u * u.conj().T
    >>>
    >>> proj_0 = e_0 * e_0.conj().T
    >>> proj_1 = e_1 * e_1.conj().T

    Then the probability of obtaining outcome :math:`0` is given by

    .. math::
        \langle P_0, \rho \rangle = \frac{1}{3}.

    >>> measure(proj_0, rho)
    0.3333333333333334

    Similarly, the probability of obtaining outcome math:`1` is given by

    .. math::
        \langle P_1, \rho \rangle = \frac{2}{3}.

    >>> measure(proj_1, rho)
    0.6666666666666666

    References
    ==========
    .. [WikMeas] Wikipedia: Measurement in quantum mechanics
        https://en.wikipedia.org/wiki/Measurement_in_quantum_mechanics

    :param measurement: The measurement to apply.
    :param state: The state to apply the measurement to.
    :return: Returns the probability of obtaining a given outcome after applying
             the variable :code:`measurement` to the variable :code:`state`.
    """
    return float(np.trace(measurement.conj().T * state))
