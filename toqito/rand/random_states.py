import numpy as np
from qiskit.quantum_info.states.random import random_statevector

def random_states(n: int, d: int) -> list[np.ndarray]:
    """
    Generate a list of random quantum states.

    Creates `n` random quantum states, each of dimension `d`. These states are generated using 
    Qiskit's `random_statevector` function, ensuring that they are valid quantum states 
    distributed according to the Haar measure.

    Examples
    ==========

    Generate three random quantum states each of dimension 4.

    >>> from toqito.rand import random_states
    >>> states = random_states(3, 4)
    >>> len(states)
    3
    >>> states[0].shape
    (4, 1)

    Verify that each state is a valid quantum state using the `is_pure_state` function from 
    `toqito`.

    >>> from toqito.state_props import is_pure_state
    >>> all(is_pure_state(state) for state in states)
    True

    Parameters
    ==========
    - n (int): The number of random states to generate.
    - d (int): The dimension of each quantum state.

    Returns
    ==========
    - list[numpy.ndarray]: A list of `n` numpy arrays. Each array is a d-dimensional quantum 
      state represented as a column vector.

    """
    return [random_statevector(d).data.reshape(-1, 1) for _ in range(n)]
