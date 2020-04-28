"""Computes the sub-fidelity of two density matrices."""

import numpy as np


def sub_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Compute the sub fidelity of two density matrices [MPHUZSub08]_.

    The sub-fidelity is a measure of similarity between density operators.
    It is defined as

    .. math::
        E(\rho, \sigma) = \text{Tr}(\rho \sigma) + \sqrt{2
        \left[ \text{Tr}(\rho \sigma)^2 - \text{Tr}(\rho \sigma \rho \sigma)
        \right]},

    where :math:`\sigma` and :math:`\rho` are density matrices. The
    sub-fidelity serves as an lower bound for the fidelity.

    Examples
    ==========

    Consider the following pair of states

    .. math::
        \rho = \frac{3}{4}|0\rangle \langle 0| +
               \frac{1}{4}|1 \rangle \langle 1|
        \sigma = \frac{1}{8}|0 \rangle \langle 0| +
                 \frac{7}{8}|1 \rangle \langle 1|.

    Calculating the fidelity between the states :math:`\rho` and :math:`\sigma`
    as :math:`F(\rho, \sigma) \approx 0.774`. This can be observed in `toqito`
    as

    >>> from toqito.core.ket import ket
    >>> from toqito.states.distance.fidelity import fidelity
    >>> e_0, e_1 = ket(2, 0), ket(2, 1)
    >>> rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    >>> sigma = 1/8 * e_0 * e_0.conj().T + 7/8 * e_1 * e_1.conj().T
    >>> print(fidelity(rho, sigma))
    0.77389339119464

    As the sub-fidelity is a lower bound on the fidelity, that is
    :math:`E(\rho, \sigma) \leq F(\rho, \sigma)`, we can use `toqito` to observe
    that :math:`E(\rho, \sigma) \approx 0.599\leq F(\rho, \sigma \approx 0.774`.

    >>> from toqito.core.ket import ket
    >>> from toqito.states.distance.sub_fidelity import sub_fidelity
    >>> e_0, e_1 = ket(2, 0), ket(2, 1)
    >>> rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    >>> sigma = 1/8 * e_0 * e_0.conj().T + 7/8 * e_1 * e_1.conj().T
    >>> print(sub_fidelity(rho, sigma))
    0.5989109809347399

    References
    ==========
    .. [MPHUZSub08] J. A. Miszczak, Z. Puchała, P. Horodecki, A. Uhlmann, K. Życzkowski
        "Sub--and super--fidelity as bounds for quantum fidelity."
        arXiv preprint arXiv:0805.2037 (2008).
        https://arxiv.org/abs/0805.2037

    :param rho: Density matrix.
    :param sigma: Density matrix.
    :return: The sub-fidelity between `rho` and `sigma`.
    """
    # Perform some error checking.
    if not np.all(rho.shape == sigma.shape):
        raise ValueError(
            "InvalidDim: `rho` and `sigma` must be matrices of the same size."
        )
    if rho.shape[0] != rho.shape[1]:
        raise ValueError("InvalidDim: `rho` and `sigma` must be square.")

    return np.real(
        np.trace(rho * sigma)
        + np.sqrt(
            2 * (np.trace(rho * sigma) ** 2 - np.trace(rho * sigma * rho * sigma))
        )
    )
