"""Smolin state."""

import numpy as np

from toqito.states.generalized_smolin import generalized_smolin


def smolin() -> np.ndarray:
    r"""Produce the four-qubit Smolin state [@smolin2001four].

    With \(|\phi_i\rangle\), \(i = 0, \dots, 3\), the four two-qubit Bell states, the Smolin state on
    qubits \(A \otimes B \otimes C \otimes D\) is

    \[
        \rho = \frac{1}{4} \sum_{i=0}^{3} |\phi_i\rangle\langle\phi_i|_{AB} \otimes |\phi_i\rangle\langle\phi_i|_{CD}
             = \frac{1}{16} \left( \mathbb{I} + X^{\otimes 4} + Y^{\otimes 4} + Z^{\otimes 4} \right).
    \]

    It is separable across each of the three 2|2 cuts, so no two of the parties can be brought to share
    entanglement by LOCC, and it has negative partial transpose across each 1|3 cut, so it is entangled.
    As soon as two parties act jointly they can Bell-measure their pair and broadcast two classical bits,
    unlocking a Bell pair for the other two: the state is bound entangled with respect to the four-party
    partition, and unlockable. It is the four-qubit member of `toqito.states.generalized_smolin`.

    Returns:
        The Smolin state as a \(16 \times 16\) matrix.

    Examples:
        Across the 2|2 cut \(AB|CD\) the state is PPT, across the 1|3 cut \(A|BCD\) it is not.

        ```python exec="1" source="above" result="text"
        from toqito.states import smolin
        from toqito.state_props import is_ppt
        rho = smolin()
        print(is_ppt(rho, dim=[4, 4]), is_ppt(rho, dim=[2, 8]))
        ```

    """
    return generalized_smolin(4)
