"""Generate random quantum states using Qiskit."""

import numpy as np
from qiskit.quantum_info.states.random import random_statevector


def random_states(n: int, d: int) -> list[np.ndarray]:
    r"""Generate a list of random quantum states.

    This function utilizes Qiskit's `random_statevector` to generate a list of quantum states,
    each of a specified dimension. The states are valid quantum states distributed according
    to the Haar measure.

    Examples
    ==========
    Generating three quantum states each of dimension 4.

    >>> from toqito.rand import random_states
    >>> states = random_states(3, 4)
    >>> len(states)
    3
    >>> states[0].shape
    (4, 1)
    >>> states  # doctest: +SKIP
    [array([[-0.2150583 +3.12920500e-01j],
           [-0.45427289-2.42799455e-01j],
           [ 0.34457387-3.20987030e-05j],
           [ 0.47739088-4.93844159e-01j]]), array([[ 0.05596192-0.40459234j],
           [-0.8497132 +0.06357884j],
           [-0.22808131-0.16261183j],
           [-0.16047978+0.05386145j]]), array([[ 0.12592373+0.00508266j],
           [-0.71527467+0.41425908j],
           [-0.27852449+0.39980357j],
           [ 0.17033502+0.18562365j]])]



    :param n: int
        The number of random states to generate.
    :param d: int
        The dimension of each quantum state.

    :return: list[numpy.ndarray]
        A list of `n` numpy arrays, each representing a d-dimensional quantum state as a
        column vector.

    """
    return [random_statevector(d).data.reshape(-1, 1) for _ in range(n)]
