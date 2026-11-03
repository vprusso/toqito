"""Generalized Smolin states."""

import numpy as np

from toqito.matrices import pauli


def generalized_smolin(num_qubits: int) -> np.ndarray:
    r"""Produce a generalized Smolin state [@augusiak2006generalized].

    For an even number \(2n\) of qubits the generalized Smolin state is

    \[
        \rho_{2n} = \frac{1}{2^{2n}} \left( \mathbb{I}^{\otimes 2n}
            + (-1)^n \sum_{\sigma \in \{X, Y, Z\}} \sigma^{\otimes 2n} \right),
    \]

    where \(\sigma^{\otimes 2n}\) applies the same Pauli operator to every qubit. The three Pauli strings
    commute and multiply to \((XYZ)^{\otimes 2n} = (-1)^n \mathbb{I}\), so their joint eigenvalues
    \((s_X, s_Y, s_Z)\) are the sign triples with product \((-1)^n\), and the coefficient \((-1)^n\) is
    what keeps every eigenvalue \(\big(1 + (-1)^n (s_X + s_Y + s_Z)\big)/2^{2n}\) non-negative.

    For \(2n = 4\) this is the Smolin state [@smolin2001four]. For every even \(2n \geq 4\) the state is
    separable across every cut that puts an even number of qubits on each side and has negative partial
    transpose across every cut with an odd number on each side, so it is bound entangled with respect to
    the fully separated partition of its qubits while no two groups of parties can distill entanglement
    from it.

    Args:
        num_qubits: The total number of qubits, a positive even integer.

    Returns:
        The generalized Smolin state on `num_qubits` qubits as a \(2^{2n} \times 2^{2n}\) matrix.

    Raises:
        ValueError: If `num_qubits` is not a positive even integer.

    Examples:
        The six-qubit state is PPT across the 2|4 cut and NPT across the 3|3 cut.

        ```python exec="1" source="above" result="text"
        from toqito.states import generalized_smolin
        from toqito.state_props import is_ppt
        rho = generalized_smolin(6)
        print(is_ppt(rho, dim=[4, 16]), is_ppt(rho, dim=[8, 8]))
        ```

    """
    if num_qubits <= 0 or num_qubits % 2:
        raise ValueError("The number of qubits must be a positive even integer.")
    sign = (-1) ** (num_qubits // 2)
    rho = np.identity(2**num_qubits, dtype=complex)
    for idx in (1, 2, 3):
        rho += sign * pauli([idx] * num_qubits)
    return rho / 2**num_qubits
