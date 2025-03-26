"""Calculate the epistemic overlap of quantum states."""

import numpy as np
from typing import List, Union
from itertools import product


def common_epistemic_overlap(density_matrices: List[np.ndarray],dim: Union[int, List[int], np.ndarray] = None) -> float:
    r"""Compute the epistemic overlap :cite:`Campos_2024_Epistemic`.

    For a set of quantum states :math:`\{\rho_i\}`, the epistemic overlap is defined as:

    .. math::
        \omega_E(\rho_1,\ldots,\rho_n) = \int \min_{\lambda\in\Lambda} 
        (\mu(\lambda|\rho_1), \ldots, \mu(\lambda|\rho_n)) d\lambda

    where :math:`\mu(\lambda|\rho_i)` are epistemic distributions associated with each state.

    The computation follows the framework using phase point operators defined for:
    - Qubits (d=2) with stabilizer vertices
    - Odd prime dimensional qudits using symplectic phase space methods

    Examples
    ==========

    Computing overlap for identical qubit states:

    >>> from toqito.states import bell
    >>> rho0 = bell(0) @ bell(0).conj().T
    >>> round(common_epistemic_overlap([rho0, rho0]),4)
    1.0

    Orthogonal qubit states:

    >>> rho1 = bell(1) @ bell(1).conj().T
    >>> round(common_epistemic_overlap([rho0, rho1]),4)
    0.0

    Maximally mixed qutrit state with pure state:

    >>> d = 3
    >>> rho_mixed = np.eye(d)/d
    >>> rho_pure = np.outer([1,0,0], [1,0,0])
    >>> round(common_epistemic_overlap([rho_mixed, rho_pure]),4 ) # doctest: +ELLIPSIS
    0.3333

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param density_matrices: List of density matrices to compute overlap for
    :param dim: Optional dimension specification for composite systems
    :return: Epistemic overlap value between 0 and 1
    :raises ValueError: For invalid inputs or unsupported dimensions
    """

    if not density_matrices:
        raise ValueError("Input list of density matrices cannot be empty")
    first_dim = density_matrices[0].shape[0]
    if dim is None:
        dim = first_dim
    if any(rho.shape[0] != first_dim for rho in density_matrices):
        raise ValueError("All density matrices must have consistent dimensions")
    vertices = _generate_phase_point_operators(first_dim)
    distributions = [_epistemic_distribution(rho, vertices) for rho in density_matrices]
    return np.sum(np.min(distributions, axis=0))


def _epistemic_distribution(rho: np.ndarray, vertices: List[np.ndarray]) -> np.ndarray:
    """Compute normalized epistemic distribution for a density matrix.
    
    :param rho: Input density matrix
    :param vertices: Precomputed phase point operators
    :return: Probability vector matching the order of vertices
    """
    probabilities = []
    for A in vertices:
        prob = np.real(np.trace(rho @ A))
        probabilities.append(prob) 
        
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("Invalid distribution (all probabilities zero)")
        
    return np.array(probabilities) / total


def _generate_phase_point_operators(d: int) -> List[np.ndarray]:
    """Generate phase point operators for dimension d via prime factorization."""
    if d == 2 or (d % 2 == 1 and _is_prime(d)):
        if d == 2:
            return _qubit_phase_operators()
        return _qudit_phase_operators(d)
    factors = _prime_factors(d)
    for p in factors:
        if p != 2 and (p % 2 == 0 or not _is_prime(p)):
            raise ValueError(f"Dimension {d} has invalid prime factor {p}")
    sub_operators = [_generate_phase_point_operators(p) for p in factors]
    vertices = []
    for ops in product(*sub_operators):
        current_op = ops[0]
        for op in ops[1:]:
            current_op = np.kron(current_op, op)
        vertices.append(current_op)
    
    return vertices


def _qubit_phase_operators() -> List[np.ndarray]:
    """Generate qubit phase point operators."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    vertices = []
    signs = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    for sx, sy, sz in signs:
        A = (I + sx*X + sy*Y + sz*Z) / 2
        vertices.append(A)
    return vertices


def _qudit_phase_operators(d: int) -> List[np.ndarray]:
    """Generate qudit phase point operators for odd prime dimensions."""
    X = _generalized_pauli_X(d)
    Z = _generalized_pauli_Z(d)
    D = _parity_operator(d)
    inv_2 = pow(2, -1, d)  
    omega = np.exp(2j * np.pi / d)
    vertices = []
    for q, p in product(range(d), repeat=2):
        phase = omega**((q * p * inv_2) % d)
        Xq = np.linalg.matrix_power(X, q)
        Zp = np.linalg.matrix_power(Z, p)
        A = (phase * Xq @ Zp @ D) / d
        vertices.append(A)
    return vertices


def _generalized_pauli_X(d: int) -> np.ndarray:
    """Generalized Pauli X (shift) operator."""
    X = np.zeros((d, d), dtype=complex)
    for i in range(d):
        X[i, (i+1) % d] = 1
    return X


def _generalized_pauli_Z(d: int) -> np.ndarray:
    """Generalized Pauli Z (clock) operator."""
    omega = np.exp(2j * np.pi / d)
    Z = np.zeros((d, d), dtype=complex)
    for i in range(d):
        Z[i, i] = omega**i
    return Z


def _parity_operator(d: int) -> np.ndarray:
    """Parity operator (|j> → |-j mod d>)."""
    D = np.zeros((d, d), dtype=complex)
    for j in range(d):
        D[j, (-j) % d] = 1
    return D


def _is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def _prime_factors(n: int) -> List[int]:
    """Prime factorization of n."""
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n = n // 2
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n = n // i
        i += 2
    if n > 2:
        factors.append(n)
    return factors