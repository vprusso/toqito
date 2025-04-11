"""Calculate the common epistemic overlap of quantum states."""

from itertools import product

import numpy as np

from toqito.matrix_ops import calculate_vector_matrix_dimension
from toqito.matrix_ops.vec import vec


def common_epistemic_overlap(states: list[np.ndarray]) -> float:
    r"""Compute the epistemic overlap :cite:`Sagnik_2024_Epistemic`.

    For a set of quantum states :math:`\{\rho_i\}` , the epistemic overlap represents the common region
    where the probability distributions of these quantum states overlap in the underlying ontic state space.
    In Einstein's epistemic model of quantum mechanics,
    this overlap quantifies the extent to which different quantum states
    share the same underlying physical reality.
    Mathematically, it is defined as:

    .. math::
        \omega_E(\rho_1,\ldots,\rho_n) = \int \min_{\lambda\in\Lambda}
        (\mu(\lambda|\rho_1), \ldots, \mu(\lambda|\rho_n)) d\lambda

    where :math:`\Lambda` is set of all ontic states and
    :math:`\mu(\lambda|\rho)` is the epistemic state associated with
    the quantum state :math:`\rho` in einstein's epistemic model.

    This function accepts both state vectors and density matrices as input.

    Examples
    ==========
    State vector inputs:

    >>> from toqito.state_props import common_epistemic_overlap
    >>> from toqito.states import bell
    >>> round(common_epistemic_overlap([bell(0), bell(2)]),4)
    np.float64(0.0)

    Mixed state inputs:

    >>> import numpy as np
    >>> from toqito.state_props import common_epistemic_overlap
    >>> round(common_epistemic_overlap([np.eye(2)/2, np.eye(2)/2]),4)
    np.float64(1.0)

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
        dims = calculate_vector_matrix_dimension(state)
        if state.ndim == 1 or (state.ndim == 2 and (state.shape[0] == 1 or state.shape[1] == 1)):
            v = vec(state)
            density_matrices.append(np.outer(v, v.conj()))
        else:
            density_matrices.append(state)
    dims = [dm.shape[0] for dm in density_matrices]
    if len(set(dims)) > 1:
        raise ValueError("All states must have consistent dimension")
    d = dims[0]

    # Generate the lambda space (ontic states)
    lambda_space = list(product(range(d), repeat=2))

    # Calculate the epistemic distributions for each density matrix
    distributions = []
    for rho in density_matrices:
        distribution = {}
        total_prob = 0.0

        # calculate  probabilities
        for lambda_point in lambda_space:
            # Construct the state vector |ℓλ⟩ associated with lambda_point
            l_lambda = np.zeros(d, dtype=complex)
            index = lambda_point[0]
            l_lambda[index] = 1.0

            # Calculate p(λ|ρ) = Tr(ρ|ℓλ⟩⟨ℓλ|)
            l_lambda_proj = np.outer(l_lambda, np.conjugate(l_lambda))
            prob = np.real(np.trace(np.matmul(rho, l_lambda_proj)))
            distribution[lambda_point] = prob
            total_prob += prob

        # normalize the distribution
        if total_prob > 0:
            for lambda_point in lambda_space:
                distribution[lambda_point] /= total_prob
        distributions.append(distribution)

    # Calculate the common epistemic overlap
    min_probabilities = {}
    for lambda_point in lambda_space:
        min_probabilities[lambda_point] = min(dist.get(lambda_point, 0) for dist in distributions)
    overlap = sum(min_probabilities.values())
    return overlap
