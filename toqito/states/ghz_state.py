import numpy as np
from typing import List
from scipy.sparse import lil_matrix


def ghz_state(dim: int,
              num_qubits: int,
              coeff: List[int] = None) -> np.ndarray:
    """
    Generates a (generalized) GHZ state.
    :param dim: The local dimension.
    :param num_qubits: The number of parties (qubits/qudits)
    :param coeff: (default [1, 1, ..., 1])/sqrt(DIM):
                  a 1-by-DIM vector of coefficients.
    :returns: Numpy vector array as GHZ state.

    Returns a NUM_QUBITS-partite GHZ state acting on DIM local dimensions,
    described in [1]. For example, ghz_state(2, 3) returns the standard
    3-qubit GHZ state on qubits. The output of this function is sparse.

    For a system of NUM_QUBITS qubits (i.e., DIM = 2), the GHZ state can be
    written as 
    |GHZ> = (|0>^{\otimes NUM_QUBITS} + |1>^{\otimes NUM_QUBITS})/sqrt(2)

    Reference:
    [1] Going beyond Bell's theorem.
        D. Greenberger and M. Horne and A. Zeilinger.
        E-print: [quant-ph] arXiv:0712.0921. 2007.
    """
    if coeff is None:
        coeff = np.ones((dim))/np.sqrt(dim)

    # Error checking:
    if dim < 2:
        raise ValueError("InvalidDim: DIM must be at least 2.")
    elif num_qubits < 2:
        raise ValueError("InvalidNumQubits: NUM_QUBITS must be at least 2.")
    elif len(coeff) != dim:
        raise ValueError("InvalidCoeff: COEFF must be a vector of length equal to DIM.")

    # Construct the state (and do it in a way that is less memory-intensive
    # than naively tensoring things together.
    dim_sum = 1
    for i in range(1, num_qubits):
        dim_sum += dim**i

    ghz_state = lil_matrix((dim**num_qubits, 1))
    for i in range(1, dim+1):
        ghz_state[(i-1)*dim_sum] = coeff[i-1]
    return ghz_state
