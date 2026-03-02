"""Generalized w-state is an entangled quantum state of `n` qubits.

This state refers to the quantum superposition in which one of the qubits is in an excited state and others are in
the ground state.
"""

import numpy as np
from scipy.sparse import csr_array


def w_state(num_qubits: int, coeff: list[int] | None = None) -> np.ndarray:
    r"""Produce a W-state [@dur2000three].

    Returns the W-state described in [@dur2000three]. The W-state on `num_qubits` qubits is defined by:

    \[
        |W \rangle = \frac{1}{\sqrt{num\_qubits}}
        \left(|100 \ldots 0 \rangle + |010 \ldots 0 \rangle + \ldots +
        |000 \ldots 1 \rangle \right).
    \]

    Examples:
        Using `|toqito⟩`, we can generate the \(3\)-qubit W-state

        \[
            |W_3 \rangle = \frac{1}{\sqrt{3}} \left( |100\rangle + |010 \rangle +
            |001 \rangle \right)
        \]

        as follows.

        ```python exec="1" source="above"
        from toqito.states import w_state
        print(w_state(3))
        ```

        We may also generate a generalized \(W\)-state. For instance, here is a \(4\)-dimensional \(W\)-state

        \[
            \frac{1}{\sqrt{30}} \left( |1000 \rangle + 2|0100 \rangle + 3|0010
            \rangle + 4 |0001 \rangle \right).
        \]

        We can generate this state in `|toqito⟩` as

        ```python exec="1" source="above"
        from toqito.states import w_state
        import numpy as np
        coeffs = np.array([1, 2, 3, 4]) / np.sqrt(30)
        print(w_state(4, coeffs))
        ```

    Raises:
        ValueError: The number of qubits must be greater than or equal to 1.

    Args:
        num_qubits: An integer representing the number of qubits.
        coeff: default is `[1, 1, ..., 1]/sqrt(num_qubits)`: a 1-by-`num_qubts` vector of coefficients.

    """
    if num_qubits < 2:
        raise ValueError("InvalidNumQubits: `num_qubits` must be at least 2.")
    if coeff is None:
        coeff_arr = np.ones(num_qubits)
    else:
        coeff_arr = np.array(coeff)
    if len(coeff_arr) != num_qubits:
        raise ValueError("InvalidCoeff: The variable `coeff` must be a vector of length equal to `num_qubits`.")

    # Normalize coefficients if necessary.
    norm = np.linalg.norm(coeff_arr)
    if not np.isclose(norm, 1.0):
        coeff_arr = coeff_arr / norm

    # Initialize a state vector of appropriate size.
    ret_w_state = csr_array((2**num_qubits, 1)).toarray()
    # Fill the vector so that the state has the single excitation distributed according to coeff.
    # Note: The ordering assumes that the binary representation corresponds to qubits in little-endian order.
    for i in range(num_qubits):
        # The position for an excitation on qubit i is at index 2**i.
        # We assign the coefficient to the position corresponding to an excitation in that qubit.
        ret_w_state[2**i] = coeff_arr[num_qubits - i - 1]
    return np.around(ret_w_state, 4)
