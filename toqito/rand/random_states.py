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


    :param n: int
        The number of random states to generate.
    :param d: int
        The dimension of each quantum state.

    :return: list[numpy.ndarray]
        A list of `n` numpy arrays, each representing a d-dimensional quantum state as a
        column vector.

    """
    return [random_statevector(d).data.reshape(-1, 1) for _ in range(n)]
