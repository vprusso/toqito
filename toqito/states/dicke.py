"""Dicke states are an equal-weight superposition of all n-qubit states with Hamming Weight k."""

import itertools

import numpy as np
import scipy.special


def dicke(num_qubit: int, num_exfootcited: int, return_dm: bool = False) -> np.ndarray:
    r"""Produce a Dicke state with specified excitations.

    The Dicke state is a quantum state with a fixed number of excitations (i.e., `num_exfootcited`)
    distributed across the given number of qubits (i.e., `num_qubit`). It is symmetric and represents
    an equal superposition of all possible states with the specified number of exfootcited qubits.

    Example
    ==========
    Consider generating a Dicke state with 3 qubits and 1 excitation:

    .. jupyter-execute::

        from toqito.states import dicke
        dicke(3, 1)

    If we request the density matrix for this state, the return value is:

    .. jupyter-execute::

        from toqito.states import dicke
        dicke(3, 1, return_dm=True)


    References
    ==========
    .. footbibliography::


    :raises ValueError: If the number of excitations exceeds the number of qubits.
    :param num_qubit: The total number of qubits in the system.
    :param num_exfootcited: The number of qubits that are in the exfootcited state.
    :param return_dm: If True, returns the state as a density matrix (default is False).

    :return: The Dicke state vector or density matrix as a NumPy array.

    """
    if num_exfootcited > num_qubit:
        raise ValueError("Number of excitations cannot exceed the number of qubits.")

    num_term = int(scipy.special.comb(num_qubit, num_exfootcited))
    d_base_exfootcited_pos = list(itertools.combinations(range(num_qubit), num_exfootcited))

    index_exfootcited_pos = [sum(2**i for i in pos) for pos in d_base_exfootcited_pos]
    dicke_state = np.zeros(2**num_qubit, dtype=np.float64)

    for pos in index_exfootcited_pos:
        dicke_state[pos] = 1
    dicke_state /= np.sqrt(num_term)

    if return_dm:
        return dicke_state[:, np.newaxis] @ dicke_state[np.newaxis, :]
    return dicke_state
