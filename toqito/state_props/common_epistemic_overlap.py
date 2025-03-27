"""Calculate the common epistemic overlap of quantum states."""

from functools import reduce
from itertools import product
from typing import List, Union

import numpy as np


def common_epistemic_overlap(
    states: List[np.ndarray], dim: Union[int, List[int], np.ndarray] = None
) -> float:
    r"""Compute the epistemic overlap :cite:`Campos_2024_Epistemic`.

    For a set of quantum states :math:`\{\rho_i\}`, the epistemic overlap is defined as:

    .. math::
        \omega_E(\rho_1,\ldots,\rho_n) = \int \min_{\lambda\in\Lambda}
        (\mu(\lambda|\rho_1), \ldots, \mu(\lambda|\rho_n)) d\lambda

    Accepts both state vectors and density matrices as input.

    Examples
    ==========
    State vector inputs:

    >>> from toqito.state_props import common_epistemic_overlap
    >>> from toqito.states import bell
    >>> psi0 = bell(0)
    >>> psi1 = bell(1)
    >>> common_epistemic_overlap([psi0, psi1])
    0.0

    Mixed state inputs:

    >>> from toqito.state_props import common_epistemic_overlap
    >>> rho_mixed = np.eye(2)/2
    >>> common_epistemic_overlap([rho_mixed, rho_mixed])
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
        if _is_state_vector(state):
            density_matrices.append(_vector_to_density_matrix(state))
        else:
            if state.shape[0] != state.shape[1]:
                raise ValueError("Density matrices must be square")
            density_matrices.append(state)
    dims = [dm.shape[0] for dm in density_matrices]
    if len(set(dims)) > 1:
        raise ValueError("All states must have consistent dimension")
    vertices = _generate_phase_point_operators(dims[0])
    distributions = [_epistemic_distribution(dm, vertices) for dm in density_matrices]
    return np.sum(np.min(distributions, axis=0))


def _is_state_vector(state: np.ndarray) -> bool:
    """Check if input is a state vector (1D or column/row vector).

    :param state: Input state to check
    :return: True if state is a vector, False otherwise

    """
    if state.ndim == 1:
        return True
    return state.ndim == 2 and (state.shape[0] == 1 or state.shape[1] == 1)


def _vector_to_density_matrix(state: np.ndarray) -> np.ndarray:
    """Convert state vector to density matrix.

    :param state: Input state vector
    :return: Corresponding density matrix

    """
    vector = state.reshape(-1)  # Flatten to 1D array
    return np.outer(vector, vector.conj())


def _epistemic_distribution(rho: np.ndarray, vertices: List[np.ndarray]) -> np.ndarray:
    """Compute normalized epistemic distribution for a density matrix.

    :param rho: Input density matrix
    :param vertices: Precomputed phase point operators
    :return: Normalized probability vector matching the order of vertices

    """
    probabilities = []
    for A in vertices:
        prob = np.real(np.trace(rho @ A))
        probabilities.append(prob)
    total = sum(probabilities)
    return np.array(probabilities) / total


def _generate_phase_point_operators(d: int) -> List[np.ndarray]:
    """Generate phase point operators for dimension d via prime factorization.

    :param d: Dimension of the quantum system
    :return: List of phase point operators
    :raises ValueError: If dimension has invalid prime factors

    """
    if d == 2 or (d % 2 == 1 and _is_prime(d)):
        return _qubit_phase_operators() if d == 2 else _qudit_phase_operators(d)

    factors = _prime_factors(d)
    sub_ops = [_generate_phase_point_operators(p) for p in factors]
    return [reduce(np.kron, ops) for ops in product(*sub_ops)]


def _qubit_phase_operators() -> List[np.ndarray]:
    """Generate qubit phase point operators.

    :return: List of 4 qubit phase point operators

    """
    Identity, X, Y, Z = (
        np.eye(2),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[1, 0], [0, -1]]),
    )
    return [
        (Identity + x * X + y * Y + z * Z) / 2
        for x, y, z in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    ]


def _qudit_phase_operators(d: int) -> List[np.ndarray]:
    """Generate qudit phase point operators for odd prime dimensions.

    :param d: Dimension of the qudit system (must be an odd prime)
    :return: List of d^2 qudit phase point operators

    """
    X, Z = _generalized_pauli_X(d), _generalized_pauli_Z(d)
    D = np.zeros((d, d))
    D[range(d), (-np.arange(d)) % d] = 1
    inv2 = pow(2, -1, d)
    omega = np.exp(2j * np.pi / d)
    return [
        (
            omega ** ((q * p * inv2) % d)
            * np.linalg.matrix_power(X, q)
            @ np.linalg.matrix_power(Z, p)
            @ D
        )
        / d
        for q, p in product(range(d), repeat=2)
    ]


def _generalized_pauli_X(d: int) -> np.ndarray:
    """Generate generalized Pauli X (shift) operator for dimension d.

    :param d: Dimension of the qudit system
    :return: d x d generalized Pauli X operator

    """
    return np.array(
        [[1 if (i + 1) % d == j else 0 for j in range(d)] for i in range(d)]
    )


def _generalized_pauli_Z(d: int) -> np.ndarray:
    """Generate generalized Pauli Z (clock) operator for dimension d.

    :param d: Dimension of the qudit system
    :return: d x d generalized Pauli Z operator

    """
    omega = np.exp(2j * np.pi / d)
    return np.diag([omega**i for i in range(d)])


def _is_prime(n: int) -> bool:
    """Check if a number is prime.

    :param n: Number to check for primality
    :return: True if n is prime, False otherwise

    """
    return n > 1 and all(n % i for i in range(2, int(np.sqrt(n)) + 1))


def _prime_factors(n: int) -> List[int]:
    """Compute the prime factorization of n.

    :param n: Number to factorize
    :return: List of prime factors of n

    """
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 2
    if n > 2:
        factors.append(n)
    return factors
