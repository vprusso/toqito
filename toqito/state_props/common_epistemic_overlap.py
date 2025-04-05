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

    This function accepts both state vectors and density matrices as input.

    Examples
    ==========
    State vector inputs:

    >>> from toqito.state_props import common_epistemic_overlap
    >>> from toqito.states import bell
    >>> common_epistemic_overlap([bell(0), bell(1)])
    np.float64(0.0)

    Mixed state inputs:

    >>> import numpy as np
    >>> from toqito.state_props import common_epistemic_overlap
    >>> common_epistemic_overlap([np.eye(2)/2, np.eye(2)/2])
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

    def prime_factors(n):
        dct = factorint(n)
        factors = []
        for p, exp in dct.items():
            factors.extend([p] * exp)
        return factors
    def qubit_phase_operators():
        I_ = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        return [(I_ + x * X + y * Y + z * Z) / 2
                for x, y, z in [(1, 1, 1),
                                (1, -1, -1),
                                (-1, 1, -1),
                                (-1, -1, 1)]]
    def qudit_phase_operators(d):
        X = gen_pauli_x(d)
        Z = gen_pauli_z(d)
        D = np.zeros((d, d))
        D[np.arange(d), (-np.arange(d)) % d] = 1
        inv2 = pow(2, -1, d)
        omega = np.exp(2j * np.pi / d)
        ops = []
        for q, p in product(range(d), repeat=2):
            op = (omega ** ((q * p * inv2) % d) *
                  np.linalg.matrix_power(X, q) @
                  np.linalg.matrix_power(Z, p) @ D) / d
            ops.append(op)
        return ops
    def generate_phase_point_operators(d):
        if d == 2 or (d % 2 == 1 and isprime(d)):
            return qubit_phase_operators() if d == 2 else qudit_phase_operators(d)
        factors = prime_factors(d)
        sub_ops = [generate_phase_point_operators(p) for p in factors]
        return [reduce(np.kron, ops) for ops in product(*sub_ops)]

    vertices = generate_phase_point_operators(d)
    distributions = []
    for dm in density_matrices:
        probs = [np.real(np.trace(dm @ A)) for A in vertices]
        total = sum(probs)
        distributions.append(np.array(probs) / total)
    overlap = max(np.sum(np.min(np.stack(distributions), axis=0)),0)
    return overlap
