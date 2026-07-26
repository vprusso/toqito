"""Generalized w-state is an entangled quantum state of `n` qubits.

This state refers to the quantum superposition in which one of the qubits is in an excited state and others are in
the ground state.
"""

import numpy as np


def w_state(num_qubits: int, coeff: list[int] | None = None) -> np.ndarray:
    r"""Produce a W-state [@dur2000three].

    Returns the W-state described in [@dur2000three]. The W-state on `num_qubits` qubits is defined by:

    \[
        |W \rangle = \frac{1}{\sqrt{num\_qubits}}
        \left(|100 \ldots 0 \rangle + |010 \ldots 0 \rangle + \ldots +
        |000 \ldots 1 \rangle \right).
    \]

    Args:
        num_qubits: An integer representing the number of qubits.
        coeff: default is `[1, 1, ..., 1]/sqrt(num_qubits)`: a 1-by-`num_qubits` vector of coefficients.

    Raises:
        ValueError: The number of qubits must be at least 2.

    Examples:
        Using `|toqito⟩`, we can generate the \(3\)-qubit W-state

        \[
            |W_3 \rangle = \frac{1}{\sqrt{3}} \left( |100\rangle + |010 \rangle +
            |001 \rangle \right)
        \]

        as follows.

        ```python exec="1" source="above" result="text"
        from toqito.states import w_state
        print(w_state(3))
        ```

        We may also generate a generalized \(W\)-state. For instance, here is a \(4\)-dimensional \(W\)-state

        \[
            \frac{1}{\sqrt{30}} \left( |1000 \rangle + 2|0100 \rangle + 3|0010
            \rangle + 4 |0001 \rangle \right).
        \]

        We can generate this state in `|toqito⟩` as

        ```python exec="1" source="above" result="text"
        from toqito.states import w_state
        import numpy as np
        coeffs = np.array([1, 2, 3, 4]) / np.sqrt(30)
        print(w_state(4, coeffs))
        ```

    """
    if num_qubits < 2:
        raise ValueError("InvalidNumQubits: `num_qubits` must be at least 2.")
    if coeff is None:
        coeff_arr = np.ones(num_qubits)
    else:
        # Flatten so column-vector input keeps working with the vectorized fill below.
        coeff_arr = np.array(coeff).ravel()
    if len(coeff_arr) != num_qubits:
        raise ValueError("InvalidCoeff: The variable `coeff` must be a vector of length equal to `num_qubits`.")

    # Normalize coefficients if necessary.
    norm = np.linalg.norm(coeff_arr)
    if not np.isclose(norm, 1.0):
        coeff_arr = coeff_arr / norm

    # Initialize a dense state vector of appropriate size, matching the coefficient dtype so complex coefficients are
    # preserved.
    ret_w_state = np.zeros((2**num_qubits, 1), dtype=coeff_arr.dtype)
    # Fill the vector so that the state has the single excitation distributed according to coeff.
    # Index 2**j carries the coefficient of qubit (num_qubits - 1 - j) in the docstring's labeling,
    # hence the reversal.
    ret_w_state[2 ** np.arange(num_qubits), 0] = coeff_arr[::-1]
    return ret_w_state
