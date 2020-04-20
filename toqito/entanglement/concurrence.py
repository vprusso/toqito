"""Computes the concurrence for a bipartite system."""
import numpy as np
from toqito.linear_algebra.matrices.pauli import pauli


def concurrence(rho: np.ndarray) -> float:
    r"""
    Calculate the concurrence of a bipartite state [WIKCON]_.

    The concurrence of a bipartite state :math:`\rho` is defined as

    .. math::
        \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4),

    where :math:`\lambda_1, \ldots, \lambda_4` are the eigenvalues in
    decreasing order of the matrix.

    Examples
    ==========

    Consider the following Bell state:

    .. math::
        u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right).

    The concurrence of the density matrix :math:`\rho = u u^*` defined by the
    vector :math:`u` is given as

    .. math::
        \mathcal{C}(\rho) \approx 1.

    The following example calculates this quantity using the `toqito` package.

    >>> import numpy as np
    >>> from toqito.core.ket import ket
    >>> from toqito.entanglement.concurrence import concurrence
    >>> e_0, e_1 = ket(2, 0), ket(2, 1)
    >>> e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)
    >>> u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
    >>> rho = u_vec * u_vec.conj().T
    >>> concurrence(rho)
    0.9999999999999998

    References
    ==========
    .. [WIKCON] Wikipedia page for concurrence (quantum computing)
       https://en.wikipedia.org/wiki/Concurrence_(quantum_computing)

    :param rho: The bipartite system specified as a matrix.
    :return: The concurrence of the bipartite state :math:`\rho`.
    """
    if rho.shape != (4, 4):
        raise ValueError(
            "InvalidDim: Concurrence is only defined for bipartite" " systems."
        )

    sigma_y = pauli("Y", False)
    sigma_y_y = np.kron(sigma_y, sigma_y)

    rho_hat = np.matmul(np.matmul(sigma_y_y, rho.conj().T), sigma_y_y)

    eig_vals = np.linalg.eigvalsh(np.matmul(rho, rho_hat))
    eig_vals = np.sort(np.sqrt(eig_vals))[::-1]
    return max(0, eig_vals[0] - eig_vals[1] - eig_vals[2] - eig_vals[3])
