"""Generates a (generalized) W-state."""
from typing import List
import numpy as np
from scipy.sparse import csr_matrix


def w_state(num_qubits: int, coeff: List[int] = None) -> np.ndarray:
    r"""
    Produce a W-state.

    Returns the W-state described in [1]. The W-state on `num_qubits` qubits is
    defined by:

    .. math::
        |W \rangle = \frac{1}{\sqrt{num\_qubits}}
        \left(|100 \ldots 0 \rangle + |010 \ldots 0 \rangle + \ldots +
        |000 \ldots 1 \rangle \right).

    References:
        [1] Three qubits can be entangled in two inequivalent ways.
        W. Dur, G. Vidal, and J. I. Cirac.
        E-print: arXiv:quant-ph/0005115, 2000.

    :param num_qubits: An integer representing the number of qubits.
    :param coeff: default is `[1, 1, ..., 1]/sqrt(num_qubits)`: a
                  1-by-`num_qubts` vector of coefficients.
    """
    if coeff is None:
        coeff = np.ones(num_qubits) / np.sqrt(num_qubits)

    if num_qubits < 2:
        raise ValueError("InvalidNumQubits: `num_qubits` must be at least 2.")
    if len(coeff) != num_qubits:
        raise ValueError(
            "InvalidCoeff: The variable `coeff` must be a vector "
            "of length equal to `num_qubits`."
        )

    ret_w_state = csr_matrix((2 ** num_qubits, 1)).toarray()

    for i in range(num_qubits):
        ret_w_state[2 ** i] = coeff[num_qubits - i - 1]

    return np.around(ret_w_state, 4)
