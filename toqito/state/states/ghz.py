"""Generates a (generalized) GHZ state."""
from typing import List

import numpy as np
import scipy as sp

from scipy.sparse import lil_matrix


def ghz(dim: int,
        num_qubits: int,
        coeff: List[int] = None) -> sp.sparse:
    """
    Generate a (generalized) GHZ state.

    Returns a `num_qubits`-partite GHZ state acting on `dim` local dimensions,
    described in [1]. For example, `ghz(2, 3)` returns the standard
    3-qubit GHZ state on qubits. The output of this function is sparse.

    For a system of `num_qubits` qubits (i.e., `dim = 2`), the GHZ state can be
    written as
    |GHZ> = (|0>^{⊗ `num_qubits`} + |1>^{⊗ `num_qubits`})/sqrt(2)

    Reference:
    [1] Going beyond Bell's theorem.
        D. Greenberger and M. Horne and A. Zeilinger.
        E-print: [quant-ph] arXiv:0712.0921. 2007.

    :param dim: The local dimension.
    :param num_qubits: The number of parties (qubits/qudits)
    :param coeff: (default `[1, 1, ..., 1])/sqrt(dim)`:
                  a 1-by-`dim` vector of coefficients.
    :returns: Numpy vector array as GHZ state.
    """
    if coeff is None:
        coeff = np.ones(dim)/np.sqrt(dim)

    # Error checking:
    if dim < 2:
        raise ValueError("InvalidDim: `dim` must be at least 2.")
    if num_qubits < 2:
        raise ValueError("InvalidNumQubits: `num_qubits` must be at least 2.")
    if len(coeff) != dim:
        raise ValueError("InvalidCoeff: The variable `coeff` must be a vector"
                         " of length equal to `dim`.")

    # Construct the state (and do it in a way that is less memory-intensive
    # than naively tensoring things together.
    dim_sum = 1
    for i in range(1, num_qubits):
        dim_sum += dim**i

    ret_ghz_state = lil_matrix((dim**num_qubits, 1))
    for i in range(1, dim+1):
        ret_ghz_state[(i-1)*dim_sum] = coeff[i-1]
    return ret_ghz_state
