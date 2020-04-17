"""Computes the super-fidelity of two density matrices."""

import numpy as np


def super_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Compute the super fidelity of two density matrices [5]_.

    The super fidelity is a measure of similarity between density operators.
    It is defined as

    .. math::
        G(\rho, \sigma) = \text{Tr}(\rho \sigma) + \sqrt{1 - \text{Tr}(\rho^2)}
        \sqrt{1 - \text{Tr}(\sigma^2)},

    where :math:`\sigma` and :math:`\rho` are density matrices. The super
    fidelity serves as an upper bound for the fidelity.

    Examples
    ==========

    Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( e_0 \otimes e_0 + e_1 \otimes e_1 \right)
        \in \mathcal{X}.

    The corresponding density matrix of :math:`u` may be calculated by:

    .. math::
        \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \text{D}(\mathcal{X}).

    In the event where we calculate the super fidelity between states that are
    identical, we should obtain the value of :math:`1`. This can be observed in
    `toqito` as follows.

    >>> from toqito.states.states.bell import bell
    >>> from toqito.states.distance.super_fidelity import super_fidelity
    >>> rho = bell(0) * bell(0).conj().T
    >>> sigma = rho
    >>>
    >>> super_fidelity(rho, sigma)
    1.0

    References
    ==========
    .. [5] J. A. Miszczak, Z. Puchała, P. Horodecki, A. Uhlmann, K. Życzkowski
        "Sub--and super--fidelity as bounds for quantum fidelity."
        arXiv preprint arXiv:0805.2037 (2008).
        https://arxiv.org/abs/0805.2037

    :param rho: Density matrix.
    :param sigma: Density matrix.
    :return: The super fidelity between `rho` and `sigma`.
    """
    # Perform some error checking.
    if not np.all(rho.shape == sigma.shape):
        raise ValueError(
            "InvalidDim: `rho` and `sigma` must be matrices of the" " same size."
        )
    if rho.shape[0] != rho.shape[1]:
        raise ValueError("InvalidDim: `rho` and `sigma` must be square.")

    return np.real(
        np.trace(rho.conj().T * sigma)
        + np.sqrt(1 - np.trace(rho.conj().T * rho))
        * np.sqrt(1 - np.trace(sigma.conj().T * sigma))
    )
