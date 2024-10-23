"""Concurrence property calculates the concurrence of a bipartite state.

The concurrence property is an entanglement measure defined for the product states of two qubits.
"""

import numpy as np

from toqito.matrices import pauli


def concurrence(rho: np.ndarray) -> float:
    r"""Calculate the concurrence of a bipartite state :cite:`WikiConcurrence`.

    The concurrence of a bipartite state :math:`\rho` is defined as

    .. math::
        \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4),

    where :math:`\lambda_1, \ldots, \lambda_4` are the square roots of the
    eigenvalues in decreasing order of the matrix

    .. math::
        \rho\tilde{\rho} = \rho \sigma_y \otimes \sigma_y \rho^* \sigma_y \otimes \sigma_y.

    Concurrence can serve as a measure of entanglement.

    Examples
    ==========

    Consider the following Bell state:

    .. math::
        u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right).

    The concurrence of the density matrix :math:`\rho = u u^*` defined by the vector :math:`u` is
    given as

    .. math::
        \mathcal{C}(\rho) \approx 1.

    The following example calculates this quantity using the :code:`toqito` package.

    >>> import numpy as np
    >>> from toqito.matrices import standard_basis
    >>> from toqito.state_props import concurrence
    >>> e_0, e_1 = standard_basis(2)
    >>> e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)
    >>> u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
    >>> rho = u_vec @ u_vec.conj().T
    >>> np.around(concurrence(rho), decimals=3)
    np.float64(1.0)

    Consider the concurrence of the following product state

    .. math::
        v = |0\rangle \otimes |1 \rangle.

    As this state has no entanglement, the concurrence is zero.

    >>> import numpy as np
    >>> from toqito.states import basis
    >>> from toqito.state_props import concurrence
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> v_vec = np.kron(e_0, e_1)
    >>> sigma = v_vec @ v_vec.conj().T
    >>> concurrence(sigma)
    0

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :raises ValueError: If system is not bipartite.
    :param rho: The bipartite system specified as a matrix.
    :return: The concurrence of the bipartite state :math:`\rho`.

    """
    if rho.shape != (4, 4):
        raise ValueError("InvalidDim: Concurrence is only defined for bipartite systems.")

    sigma_y = pauli("Y", False)
    sigma_y_y = np.kron(sigma_y, sigma_y)

    rho_tilde = sigma_y_y @ rho.conj() @ sigma_y_y

    eig_vals = np.linalg.eigvals(rho @ rho_tilde)
    eig_vals = np.sort(np.abs(np.sqrt(eig_vals)))[::-1]
    return max(0, eig_vals[0] - eig_vals[1] - eig_vals[2] - eig_vals[3])
