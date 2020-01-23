import numpy as np
from typing import List
from scipy.sparse import csr_matrix


def w_state(num_qubits: int, coeff: List[int] = None) -> np.ndarray:
    """
    Produces a W-state.
    :param num_qubits: An integer representing the number of qubits.
    :param coeff: default is [1, 1, ..., 1]/sqrt(NUM_QUBITS): a 1-by-NUM_QUBITS
                  vector of coefficients.

    Returns the W-state described in [1].

    The W-state on NUM_QUBITS qubits is defined by:
        |W> = 1/sqrt(NUM_QUBITS) * (|100...0> + |010...0> + ... + |000...1>).

    References:
    [1] Three qubits can be entangled in two inequivalent ways.
        W. Dur, G. Vidal, and J. I. Cirac. 
        E-print: arXiv:quant-ph/0005115, 2000.

    """
    if coeff is None:
        coeff = np.ones(num_qubits)/np.sqrt(num_qubits)

    if num_qubits < 2:
        raise ValueError("ValueError: NUM_QUBITS must be at least 2.")
    elif len(coeff) != num_qubits:
        raise ValueError("ValueError: COEFF must be a vector of length equal to NUM_QUBITS.")

    w_state = csr_matrix((2**num_qubits, 1)).toarray()

    for i in range(num_qubits):
        w_state[2**(i)] = coeff[num_qubits-i-1]

    return np.around(w_state, 4)
