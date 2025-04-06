"""Calculate the common epistemic overlap of quantum states."""

from functools import reduce
from itertools import product

import numpy as np
from sympy import factorint, isprime

from toqito.matrices import gen_pauli_x, gen_pauli_z
from toqito.matrix_ops.vec import vec


def common_epistemic_overlap(states, dim=None) -> float:
    r"""Compute the epistemic overlap :cite:`Sagnik_2024_Epistemic`.

    For a set of quantum states :math:`\{\rho_i\}`, the epistemic overlap is defined as:

    .. math::
        \omega_E(\rho_1,\ldots,\rho_n) = \int \min_{\lambda\in\Lambda}
        (\mu(\lambda|\rho_1), \ldots, \mu(\lambda|\rho_n)) d\lambda

    where :math:`\mu(\lambda|\rho)` is the epistemic state associated with the quantum state :math:`\rho` in einstein's epistemic model

    This function accepts both state vectors and density matrices as input.

    Examples
    ==========
    State vector inputs:

    >>> from toqito.state_props import common_epistemic_overlap
    >>> from toqito.states import bell
    >>> round(common_epistemic_overlap([bell(0), bell(2)]),4)
    0.0

    Mixed state inputs:

    >>> import numpy as np
    >>> from toqito.state_props import common_epistemic_overlap
    >>> round(common_epistemic_overlap([np.eye(2)/2, np.eye(2)/2]),4)
    1.0

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param states: List of quantum states (vectors or density matrices)
    :param dim: Optional dimension specification for composite systems
    :raises ValueError: For invalid inputs or unsupported dimensions
    :return: Common epistemic overlap value between 0 and 1

    """
    density_matrices = []
    for state in states:
        if state.ndim == 1 or (state.ndim == 2 and (state.shape[0] == 1 or state.shape[1] == 1)):
            v = vec(state)
            density_matrices.append(np.outer(v, v.conj()))
        else:
            if state.shape[0] != state.shape[1]:
                raise ValueError("Density matrices must be square")
            density_matrices.append(state)
    dims = [dm.shape[0] for dm in density_matrices]
    if len(set(dims)) > 1:
        raise ValueError("All states must have consistent dimension")
    d = dims[0]
    lambda_space = list(product(range(d), repeat=2))
    distributions = []
    for rho in density_matrices:
        distribution = {}
        total_prob = 0.0
        for lambda_point in lambda_space:
            l_lambda = np.zeros(d, dtype=complex)
            index = lambda_point[0]
            l_lambda[index] = 1.0
            l_lambda_proj = np.outer(l_lambda, np.conjugate(l_lambda))
            prob = np.real(np.trace(np.matmul(rho, l_lambda_proj)))
            distribution[lambda_point] = prob
            total_prob += prob
        if total_prob > 0:
            for lambda_point in lambda_space:
                distribution[lambda_point] /= total_prob
        distributions.append(distribution)
    min_probabilities = {}
    for lambda_point in lambda_space:
        min_probabilities[lambda_point] = min(dist.get(lambda_point, 0) for dist in distributions)
    overlap = sum(min_probabilities.values())
    return overlap
