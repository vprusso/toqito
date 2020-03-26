"""Computes the concurrence for a bipartite system."""
import numpy as np
from toqito.matrix.matrices.pauli import pauli


def concurrence(rho: np.ndarray) -> float:
    r"""
    Calculate the concurrence of a bipartite state.

    The concurrence of a bipartite state :math:`\rho` is defined as

    .. math::
        \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4),

    where :math:`\lambda_1, \ldots, \lambda_4` are the eigenvalues in
    decreasing order of the matrix.

    References:
        [1] Wikipedia page for concurrence (quantum computing)
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
